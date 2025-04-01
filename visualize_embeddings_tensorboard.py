import torch
from torch.utils.tensorboard import SummaryWriter
from vqvae import VQVAE
from utils import get_transform, get_device
from torchvision.datasets import CIFAR10
from constants import CIFAR10_DATA_ROOT
from pathlib import Path
from eval_utils import get_receptive_field_coords


def main():
    # Initialize TensorBoard writer.
    writer = SummaryWriter("logs/vqvae_embeddings")

    checkpoint_path = Path("checkpoints/vqvae_cifar10/run_2025-03-30_00-49-32/model.pth")
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize model with saved args
    device = get_device()
    model = VQVAE(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get the codebook embeddings from the vector quantizer.
    # Note: e_i_ts has shape (embedding_dim, num_embeddings).
    # We need to transpose it to have shape (num_embeddings, embedding_dim).
    codebook_embeddings = model.vq.e_i_ts.detach().cpu().t()

    metadata = [f"Embedding {i}" for i in range(codebook_embeddings.shape[0])]

    writer.add_embedding(codebook_embeddings, metadata=metadata, tag="Codebook Embeddings", global_step=0)
    print("Logged codebook embeddings to TensorBoard projector.")


    # Load one CIFAR10 image and resize to 64x64.
    transform = get_transform()
    cifar_dataset = CIFAR10(root=CIFAR10_DATA_ROOT, download=False, transform=transform)
    image, _ = cifar_dataset[0]
    image = image.unsqueeze(0).to(device)
    image = image.requires_grad_(True)

    # Pass the image through the VQVAE quantization.
    z_quantized, _, _, encoding_indices = model.quantize(image)
    # Reshape encoding_indices into a spatial grid.
    _, _, H_latent, W_latent = z_quantized.shape
    encoding_indices = encoding_indices.view(1, H_latent, W_latent)

    # Get unique embedding indices used.
    unique_indices = torch.unique(encoding_indices)
    print("Unique embedding indices in quantized representation:", unique_indices.tolist())

    # For each unique embedding, compute its receptive field patch.
    for i, emb_idx in enumerate(unique_indices):
        # Find spatial locations where this embedding is used.
        mask = (encoding_indices == emb_idx)
        indices = mask.nonzero(as_tuple=False)
        if indices.numel() == 0:
            continue

        # Choose one spatial location (e.g. the middle one) for this embedding.
        batch_idx, h, w = indices[len(indices)//2].tolist()
        scalar_value = z_quantized[batch_idx, :, h, w].sum()

        # Zero gradients before backward.
        if image.grad is not None:
            image.grad.zero_()

        # Retain graph if not the last iteration.
        retain = True if i < (len(unique_indices) - 1) else False
        scalar_value.backward(retain_graph=retain)

        # Get the gradient from the input image.
        grad_np = image.grad.detach().cpu().numpy()[0]  # Shape: (C, 64, 64)


        # # Compute receptive field coordinates based on the gradient.
        coords = get_receptive_field_coords(grad_np, threshold=1e-3)
        if coords is None:
            print(f"No significant receptive field found for embedding {int(emb_idx.item())}.")
            continue
        top, bottom, left, right = coords
        print(f"Embedding {int(emb_idx.item())} at (h={h}, w={w}) has receptive field coords: top={top}, bottom={bottom}, left={left}, right={right}.")

        # Extract the image patch corresponding to the receptive field.
        image_np = image.detach().cpu().numpy()[0]  # (C, 64, 64)
        patch = image_np[:, top:bottom+1, left:right+1]

        # Log the image patch to TensorBoard.
        writer.add_image(f"Embedding_{int(emb_idx.item())}_patch", patch, global_step=0)

    writer.close()


if __name__ == "__main__":
    main()
