# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import argparse
import numpy as np
import torch
from tqdm import tqdm
import math

from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from models.vqvae import VQVAE
from utils.torch_utils import get_transform
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils.torch_utils import get_device, fix_seed
from pathlib import Path
import itertools
from torchvision.utils import make_grid
import logging


def train(submit_config, model_kwargs, dataset_kwargs, training_kwargs, device, batch_size=256, subset_size=512):
    """Initializes and trains the VQ-VAE model."""
    # -----------------------------------------------------------------------
    # 1) SETUP
    # -----------------------------------------------------------------------
    # Setup logging
    run_dir = submit_config.get('run_dir', '.')
    writer = SummaryWriter(log_dir=Path(run_dir) / 'tensorboard')

    # Model, optimizer, loss
    model = VQVAE(**model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    beta = 0.25  # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation Learning".

    # Initialize dataset.
    workers = 2
    transform = get_transform()
    download = False

    # Prepare data
    logging.info(f"Loading train dataset from {dataset_kwargs.get('data_root')}")
    train_dataset = CIFAR10(dataset_kwargs.get('data_root'), True, transform, download=download)
    logging.info(f"Total dataset size: {len(train_dataset)}, training on subset of size: {subset_size}, batch size: {batch_size}")
    train_subset = Subset(train_dataset, range(subset_size))
    train_data_variance = np.var(train_subset.dataset.data[train_subset.indices] / 255)
    logging.info(f"Train data variance: {train_data_variance}")

    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    # Create an infinite iterator
    train_loader_iter = itertools.cycle(train_loader)

    eval_loader = DataLoader(dataset=train_subset, batch_size=32, shuffle=False, num_workers=workers, pin_memory=True)
    fixed_eval_images = next(iter(eval_loader))[0].to(device)


    # Training settings
    total_training_images = training_kwargs['total_training_images']
    eval_every = training_kwargs['eval_every']
    total_images_processed = 0

    best_train_loss = float("inf")
    best_recon_err = float("inf")
    model.train()

    # -----------------------------------------------------------------------
    # 2) TRAINING LOOP
    # -----------------------------------------------------------------------
    with tqdm(total=total_training_images, desc="Training Progress") as pbar:
        while total_images_processed <= total_training_images:
            # ---- 2.1) EVAL / LOG PASS ----
            if total_images_processed % eval_every == 0:
                model.eval()

                with torch.no_grad():
                    writer.add_histogram('distribution/codebook_embeddings', model.vq.e_i_ts.detach().cpu(), global_step=total_images_processed)

                    if total_images_processed > 0:
                        # log flat_x stats
                        flat_x = model.vq.last_flat_x       # still on GPU
                        flat_cpu = flat_x.detach().cpu()    # move off-GPU
                        model.vq.last_flat_x = None         # free GPU memory

                        writer.add_histogram(
                            "distribution/vq_pre_quantization_input",
                            flat_cpu,
                            global_step=total_images_processed,
                        )
                        writer.add_scalar("stats/flat_x_mean", flat_cpu.mean(), total_images_processed)
                        writer.add_scalar("stats/flat_x_std",  flat_cpu.std(),  total_images_processed)
                        writer.add_scalar("stats/flat_x_min",  flat_cpu.min(),  total_images_processed)
                        writer.add_scalar("stats/flat_x_max",  flat_cpu.max(),  total_images_processed)

                    # Use the fixed evaluation images for consistent comparison.
                    out_eval = model(fixed_eval_images)
                    imgs_viz = (fixed_eval_images.clamp(-0.5, 0.5) + 0.5)
                    recon_viz = (out_eval.clamp(-0.5, 0.5) + 0.5)

                    grid_orig = make_grid(imgs_viz, nrow=8)  # Adjust nrow as needed.
                    grid_recon = make_grid(recon_viz, nrow=8)

                    writer.add_image('eval/original_images', grid_orig, total_images_processed)
                    writer.add_image('eval/reconstructed_images', grid_recon, total_images_processed)

                    # Calculate and log the difference image
                    diff_image = (imgs_viz - recon_viz).abs() # Absolute difference
                    # Optional magnification - clamp to ensure it stays within [0, 1]
                    magnification_factor = 5
                    diff_viz = (diff_image * magnification_factor).clamp(0, 1)

                    grid_diff = make_grid(diff_viz, nrow=8)
                    writer.add_image('eval/difference_images', grid_diff, total_images_processed)

            # # ---- 2.2) TRAIN PASS ----
            model.train()

            try:
                train_tensors = next(train_loader_iter)
            except StopIteration:
                # This should not happen with itertools.cycle, but as a safeguard
                train_loader_iter = itertools.cycle(train_loader)
                train_tensors = next(train_loader_iter)

            optimizer.zero_grad()
            imgs = train_tensors[0].to(device)
            current_batch_size = imgs.size(0)

            out = model(imgs)
            recon_error = criterion(out, imgs) / train_data_variance
            loss = recon_error + beta * model.commitment_loss
            if not model_kwargs.get('use_ema', True):
                loss += model.dictionary_loss

            loss.backward()
            optimizer.step()

            # ---- 2.3) LOG TRAIN METRICS ----
            # scalar losses
            writer.add_scalar('train/loss', loss.item(), total_images_processed)
            writer.add_scalar('train/recon_error', recon_error.item(), total_images_processed)
            writer.add_scalar('train/commitment_loss', model.commitment_loss.item(), total_images_processed)
            if not model_kwargs.get('use_ema', True):
                writer.add_scalar('train/dictionary_loss', model.dictionary_loss.item(), total_images_processed)

            # track best loss
            if loss.item()      < best_train_loss:   best_train_loss = loss.item()
            if recon_error.item() < best_recon_err:  best_recon_err  = recon_error.item()

            # — Checkpoint best model —
            if loss.item() < best_train_loss:
                best_loss = loss.item()
                ckpt = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_kwargs': model_kwargs,
                }
                torch.save(ckpt, run_dir / "best_model.pth")
                logging.info(f"Best model checkpoint saved to {run_dir / 'best_model.pth'}")

            # increment & progress bar
            total_images_processed += current_batch_size
            pbar.update(imgs.size(0))
            pbar.set_postfix({
                "ReconErr": recon_error.item(),
                "BestRecon": best_recon_err,
                "BestLoss":  best_train_loss
            })


    hparams_to_log = {
        'num_hiddens': model_kwargs.get('num_hiddens'),
        'num_residual_hiddens': model_kwargs.get('num_residual_hiddens'),
        'embedding_dim': model_kwargs.get('embedding_dim'),
        'num_embeddings': model_kwargs.get('num_embeddings')
    }
    
    metrics_to_log = {
        'hparam/best_train_loss': best_train_loss
    }
    writer.add_hparams(hparams_to_log, metrics_to_log)
    writer.close()

    return best_train_loss