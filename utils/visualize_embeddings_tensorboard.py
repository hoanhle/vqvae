import torch
from torch.utils.tensorboard import SummaryWriter
from vqvae import VQVAE
from utils.utils import get_transform, get_device
from torchvision.datasets import CIFAR10
from utils.utils import CIFAR10_DATA_ROOT
from pathlib import Path
from eval_utils import get_receptive_field_coords
import logging
from tqdm import tqdm
from collections import defaultdict
from torchvision.utils import make_grid
from torch.nn import functional as F


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def pad_to_size(t, target_size): 
    """ Pad tensor t (shape: C x H x W) with zeros so that it matches target_size (target_H, target_W). 
    Padding is applied on the bottom and right sides. """ 
    C, H, W = t.shape 
    target_H, target_W = target_size 
    pad_bottom = target_H - H 
    pad_right = target_W - W 
    # F.pad expects pad in the order: (left, right, top, bottom) 
    padded = F.pad(t, (0, pad_right, 0, pad_bottom), mode='constant', value=0) 
    return padded


def main():
    # Initialize TensorBoard writer.
    writer = SummaryWriter("../logs/vqvae_embeddings")

    checkpoint_path = Path("../checkpoints/vqvae_cifar10/run_2025-03-30_00-49-32/model.pth")
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
    logging.debug("Logged codebook embeddings to TensorBoard projector.")


    # Load one CIFAR10 image and resize to 64x64.
    transform = get_transform()
    cifar_dataset = CIFAR10(root=CIFAR10_DATA_ROOT, download=False, transform=transform)

    # Specify the number of images to process.
    n_images = 100

    embedding_to_patches = defaultdict(list)

    for img_idx in tqdm(range(n_images)):
        image, _ = cifar_dataset[img_idx]
        image = image.unsqueeze(0).to(device)
        image = image.requires_grad_(True)

        # Pass the image through the VQVAE quantization.
        z_quantized, _, _, encoding_indices = model.quantize(image)
        # Reshape encoding_indices into a spatial grid.
        _, _, H_latent, W_latent = z_quantized.shape
        encoding_indices = encoding_indices.view(1, H_latent, W_latent)

        # Get unique embedding indices used.
        unique_indices = torch.unique(encoding_indices)
        logging.debug("Unique embedding indices in quantized representation: %s", unique_indices.tolist())

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


            coords = get_receptive_field_coords(grad_np, threshold=1e-3)
            if coords is None:
                logging.debug(f"No significant receptive field found for embedding {int(emb_idx.item())}.")
                continue
            top, bottom, left, right = coords
            logging.debug(f"Embedding {int(emb_idx.item())} at (h={h}, w={w}) has receptive field coords: top={top}, bottom={bottom}, left={left}, right={right}.")

            image_np = image.detach().cpu().numpy()[0]
            patch = image_np[:, top:bottom+1, left:right+1]

            # Log the image patch to TensorBoard.
            patch_tensor = torch.tensor(patch, dtype=torch.float)
            emb_idx_val = int(encoding_indices[0, h, w].item())
            embedding_to_patches[emb_idx_val].append(patch_tensor)

    # For each embedding, pad patches to a common size and log them as a grid.
    for emb_idx, patches in embedding_to_patches.items():
        if len(patches) == 0:
            continue

        # Determine maximum height and width among patches.
        max_H = max(p.shape[1] for p in patches)
        max_W = max(p.shape[2] for p in patches)
        target_size = (max_H, max_W)

        padded_patches = [pad_to_size(p, target_size) for p in patches]
        batch = torch.stack(padded_patches)  # shape: (N, C, H, W)
        grid = make_grid(batch, nrow=8)
        writer.add_image(f"Embedding_{emb_idx}_all_patches", grid, global_step=0)
        logging.debug(f"Logged grid for Embedding {emb_idx} with {len(patches)} patches.")

    writer.close()


if __name__ == "__main__":
    main()
