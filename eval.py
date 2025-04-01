import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from vqvae import VQVAE
from utils import save_img_tensors_as_grid
from pathlib import Path
from constants import CIFAR10_DATA_ROOT
from utils import get_transform

def main():
    # Load the model checkpoint
    checkpoint_path = Path("checkpoints/vqvae_cifar10/run_2025-03-30_00-49-32/model.pth")
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize model with saved args
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()    

    transform = get_transform()
    test_dataset = CIFAR10(CIFAR10_DATA_ROOT, False, transform, download=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=10,
    )
    
    # Generate reconstructions
    with torch.no_grad():
        test_batch = next(iter(test_loader))[0].to(device)
        out = model(test_batch)
        reconstructions = out["x_recon"]
        
        # Save original and reconstructed images
        save_dir = checkpoint_path.parent
        save_img_tensors_as_grid(test_batch, 4, save_dir / "original.png")
        save_img_tensors_as_grid(reconstructions, 4, save_dir / "reconstructed.png")

if __name__ == "__main__":
    main() 