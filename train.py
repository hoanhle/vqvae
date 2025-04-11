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
from utils.torch_utils import get_device
from pathlib import Path


def train(submit_config, model_kwargs, dataset_kwargs, training_kwargs, device):
    """Initializes and trains the VQ-VAE model."""
    # Setup logging
    run_dir = submit_config.get('run_dir', '.')
    writer = SummaryWriter(log_dir=Path(run_dir) / 'tensorboard')

    # Initialize model.
    model = VQVAE(**model_kwargs).to(device)

    # Initialize dataset.
    batch_size = 32
    workers = 10
    transform = get_transform()
    download = False
    train_dataset = CIFAR10(dataset_kwargs.get('data_root'), dataset_kwargs.get('download'), transform, download=download)
    train_data_variance = np.var(train_dataset.data / 255)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )

    # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation
    # Learning".
    beta = 0.25

    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    lr = 3e-4
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()

    # Training settings
    epochs = training_kwargs.get('epochs', 1)
    eval_every = training_kwargs.get('eval_every', 10)
   
    best_train_loss = float("inf")
    model.train()
    global_step = 0

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        total_train_loss = 0
        total_recon_error = 0
        n_train = 0
        for (batch_idx, train_tensors) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            optimizer.zero_grad()
            imgs = train_tensors[0].to(device)
            out = model(imgs)
            recon_error = criterion(out["x_recon"], imgs) / train_data_variance
            total_recon_error += recon_error.item()
            loss = recon_error + beta * out["commitment_loss"]
            if not model_kwargs.get('use_ema', True):
                loss += out["dictionary_loss"]

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1
            global_step += 1

            if ((batch_idx + 1) % eval_every) == 0:
                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
                avg_train_loss = total_train_loss / n_train
                avg_recon_error = total_recon_error / n_train
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss

                # Log metrics to TensorBoard
                writer.add_scalar('Loss/train', avg_train_loss, global_step)
                writer.add_scalar('ReconstructionError/train', avg_recon_error, global_step)
                writer.add_scalar('Loss/best_train', best_train_loss, global_step)

                print(f"total_train_loss: {avg_train_loss}")
                print(f"best_train_loss: {best_train_loss}")
                print(f"recon_error: {avg_recon_error}\n")

                total_train_loss = 0
                total_recon_error = 0
                n_train = 0

    save_path = run_dir / "model.pth"
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


def main(args):
    # Setup device
    device = get_device()

    # Setup logging and saving directory
    run_name = datetime.now().strftime(f"run_%Y-%m-%d_%H-%M-%S_hid{args.num_hiddens}_res{args.num_residual_hiddens}_emb{args.embedding_dim}_num{args.num_embeddings}")
    save_dir = Path("log/vqvae_cifar10") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    use_ema = True

    model_kwargs = {
        "in_channels": 3,
        "num_hiddens": args.num_hiddens,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": args.num_residual_hiddens,
        "embedding_dim": args.embedding_dim,
        "num_embeddings": args.num_embeddings,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }

    submit_config = {
        'run_dir': save_dir,
    }

    dataset_kwargs = {
        'data_root': '/home/leh19/datasets/cifar10/',
        'download': False,
    }

    training_kwargs = {
        'epochs': 2,
        'eval_every': 100,
    }

    # Train model.
    best_train_loss = train(
        submit_config=submit_config,
        model_kwargs=model_kwargs,
        dataset_kwargs=dataset_kwargs,
        training_kwargs=training_kwargs,
        device=device,
    )

    print(f"Training finished. Best train loss: {best_train_loss}")
    print(f"Checkpoints and logs saved in: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VQ-VAE model')
    # Hyperparameters for sweeping
    parser.add_argument('--num_hiddens', type=int, default=128, help='Number of hidden channels in Conv layers')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='Number of hidden channels in Residual blocks')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of each codebook vector')
    parser.add_argument('--num_embeddings', type=int, default=512, help='Number of codebook vectors (codebook size)')

    args = parser.parse_args()
    main(args)