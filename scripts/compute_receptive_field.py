#!/usr/bin/env python
import torch
import torchvision
from vqvae import VQVAE
import numpy as np
import matplotlib.pyplot as plt
from utils.torch_utils import fix_seed, get_device, get_transform


def compute_receptive_field_bbox(grad, threshold=1e-5):
    """
    Given a gradient of shape (C, H, W), compute the bounding box (top, bottom, left, right)
    for the region where the maximum absolute gradient across channels exceeds a threshold.
    """
    combined = np.max(np.abs(grad), axis=0)
    mask = combined > threshold
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()
    return top, bottom, left, right


def prepare_dataloader(batch_size=1):
    transform = get_transform()
    dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_model(device):
    """Load and return the VQVAE model."""
    model_kwargs = {
        "in_channels": 3,
        "num_hiddens": 256,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 256,
        "embedding_dim": 16,
        "num_embeddings": 512,
        "use_ema": True,
        "decay": 0.99,
        "epsilon": 1e-5,
        "kernel_size": 2,
        "stride": 2,
        "padding": 1,
    }
    model = VQVAE(**model_kwargs).to(device)
    model.eval()
    return model


def main():
    # Fix random seed for reproducibility.
    fix_seed(42)
    
    # Setup device and dataloader.
    device = get_device()
    dataloader = prepare_dataloader(batch_size=1)
    
    # Load the VQVAE model.
    model = load_model(device)
    
    # Fetch a single image from the dataloader.
    data_iter = iter(dataloader)
    input_image, _ = next(data_iter)
    input_image = input_image.to(device)
    input_image.requires_grad = True  # Enable gradients for receptive field analysis

    # Pass image through the encoder and pre-quantization convolution.
    z = model.pre_vq_conv(model.encoder(input_image))  # shape: (B, embedding_dim, H, W)
    _, _, H, W = z.shape

    i, j = H // 2, W // 2
    print(f"Selected middle pixel at position (i={i}, j={j}).")
    
    # Compute scalar activation from the chosen pixel (summing across channels).
    scalar_activation = z[0, :, i, j].sum()
    
    # Compute gradients with respect to the input image.
    if input_image.grad is not None:
        input_image.grad.zero_()
    scalar_activation.backward(retain_graph=True)
    grad_np = input_image.grad[0].detach().cpu().numpy()  # shape: (C, H_img, W_img)
    
    # Compute receptive field bounding box.
    bbox = compute_receptive_field_bbox(grad_np, threshold=1e-5)
    if bbox is not None:
        top, bottom, left, right = bbox
        rf_height = bottom - top + 1
        rf_width = right - left + 1
        print(f"Receptive field bounding box = {bbox}, shape = {(rf_height, rf_width)}")
    else:
        print("No significant receptive field found.")

    # For visualization: overlay the bounding box on the combined gradient map.
    combined_grad = np.max(np.abs(grad_np), axis=0)
    plt.figure(figsize=(6, 6))
    plt.imshow(combined_grad, cmap="hot")
    plt.title("Gradient Map and Receptive Field (Middle Pixel)")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
