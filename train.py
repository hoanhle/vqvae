# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import argparse
import numpy as np
import torch
from tqdm import tqdm

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from vqvae import VQVAE
from utils.torch_utils import get_transform
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils.torch_utils import get_device, fix_seed
from pathlib import Path
import itertools
from torchvision.utils import make_grid


def train(submit_config, model_kwargs, dataset_kwargs, training_kwargs, device):
    """Initializes and trains the VQ-VAE model."""
    # Setup logging
    run_dir = submit_config.get('run_dir', '.')
    writer = SummaryWriter(log_dir=Path(run_dir) / 'tensorboard')

    # Initialize model.
    model = VQVAE(**model_kwargs).to(device)

    # Initialize dataset.
    batch_size = 256
    workers = 2
    transform = get_transform()
    download = False

    print(f"Loading train dataset from {dataset_kwargs.get('data_root')}")
    train_dataset = CIFAR10(dataset_kwargs.get('data_root'), True, transform, download=download)
    train_data_variance = np.var(train_dataset.data / 255)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    # Create an infinite iterator
    train_loader_iter = itertools.cycle(train_loader)

    test_dataset = CIFAR10(dataset_kwargs.get('data_root'), False, transform, download=download)
    eval_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    fixed_eval_images = next(iter(eval_loader))[0].to(device)

    # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation
    # Learning".
    beta = 0.25

    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    lr = 3e-4
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()

    # Training settings
    total_training_images = training_kwargs['total_training_images']
    eval_every = training_kwargs['eval_every']

    best_train_loss = float("inf")
    best_recon_error = float("inf")
    model.train()
    total_images_processed = 0

    # Use tqdm for progress based on total images
    with tqdm(total=total_training_images, desc="Training Progress") as pbar:
        while total_images_processed < total_training_images:
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
            recon_error = criterion(out["x_recon"], imgs) / train_data_variance
            loss = recon_error + beta * out["commitment_loss"]
            if not model_kwargs.get('use_ema', True):
                loss += out["dictionary_loss"]

            loss.backward()
            optimizer.step()

            # Log metrics to TensorBoard
            writer.add_scalar('train/loss', loss.item(), total_images_processed)
            writer.add_scalar('train/recon_error', recon_error.item(), total_images_processed)
            writer.add_scalar('train/commitment_loss', out['commitment_loss'].item(), total_images_processed)
            if not model_kwargs.get('use_ema', True):
                writer.add_scalar('train/dictionary_loss', out['dictionary_loss'].item(), total_images_processed)

            # Track best loss
            current_loss = loss.item()
            if current_loss < best_train_loss:
                best_train_loss = current_loss

            # Track best reconstruction error
            current_recon_error = recon_error.item()
            if current_recon_error < best_recon_error:
                best_recon_error = current_recon_error

            # Check if it's time to log images
            if total_images_processed > 0 and total_images_processed % eval_every == 0:
                with torch.no_grad():
                    # Use the fixed evaluation images for consistent comparison.
                    out_eval = model(fixed_eval_images)
                    imgs_viz = (fixed_eval_images.clamp(-0.5, 0.5) + 0.5)
                    recon_viz = (out_eval["x_recon"].clamp(-0.5, 0.5) + 0.5)

                    grid_orig = make_grid(imgs_viz, nrow=4)  # Adjust nrow as needed.
                    grid_recon = make_grid(recon_viz, nrow=4)

                    writer.add_image('eval/original_images', grid_orig, total_images_processed)
                    writer.add_image('eval/reconstructed_images', grid_recon, total_images_processed)

                    
            total_images_processed += current_batch_size

            # Update progress bar
            pbar.update(current_batch_size)
            pbar.set_postfix({"ReconErr": current_recon_error, "BestReconErr": best_recon_error, "BestLoss": best_train_loss})

    save_path = Path(run_dir) / "model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }, save_path)
    print(f"Model saved to {save_path}")

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