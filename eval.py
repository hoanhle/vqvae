import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from models.vqvae import VQVAE
from utils.torch_utils import save_img_tensors_as_grid
from pathlib import Path
from utils.constants import CIFAR10_DATA_ROOT
from utils.torch_utils import get_transform, get_device

def main():
    # Load the model checkpoint
    checkpoint_path = Path("log/vqvae_cifar10/run_2025-04-15_13-38-30_hid256_res256_emb32_num32768/model.pth")
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize model with saved args
    device = get_device()
    model = VQVAE(**checkpoint['model_kwargs']).to(device)
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