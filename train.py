# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import numpy as np
import torch
from tqdm import tqdm

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from vqvae import VQVAE
from utils.utils import get_transform
from datetime import datetime
from pathlib import Path



def main():
    # Initialize model.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    use_ema = True
    model_args = {
        "in_channels": 3,
        "num_hiddens": 128,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    model = VQVAE(**model_args).to(device)

    # Initialize dataset.
    batch_size = 32
    workers = 10
    transform = get_transform()
    data_root = "./data/cifar10"
    download = False
    train_dataset = CIFAR10(data_root, True, transform, download=download)
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

    # Train model.
    epochs = 1
    eval_every = 1000
    best_train_loss = float("inf")
    model.train()
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
            if not use_ema:
                loss += out["dictionary_loss"]

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

            if ((batch_idx + 1) % eval_every) == 0:
                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
                total_train_loss /= n_train
                if total_train_loss < best_train_loss:
                    best_train_loss = total_train_loss

                print(f"total_train_loss: {total_train_loss}")
                print(f"best_train_loss: {best_train_loss}")
                print(f"recon_error: {total_recon_error / n_train}\n")

                total_train_loss = 0
                total_recon_error = 0
                n_train = 0
    
    run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    save_dir = Path("checkpoints/vqvae_cifar10") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_args': model_args,
    }, save_path)

if __name__ == "__main__":
    main()