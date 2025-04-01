import torch
from torch.utils.tensorboard import SummaryWriter
from vqvae import VQVAE


def main():
    # Initialize TensorBoard writer.
    writer = SummaryWriter("logs/vqvae_embeddings")

    # Define model arguments (same as your vqvae main block)
    model_args = {
        "in_channels": 3,
        "num_hiddens": 128,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "use_ema": True,
        "decay": 0.99,
        "epsilon": 1e-5,
    }

    model = VQVAE(**model_args)
    model.eval()

    # Get the codebook embeddings from the vector quantizer.
    # Note: e_i_ts has shape (embedding_dim, num_embeddings).
    # We need to transpose it to have shape (num_embeddings, embedding_dim).
    codebook_embeddings = model.vq.e_i_ts.detach().cpu().t()

    metadata = [f"Embedding {i}" for i in range(codebook_embeddings.shape[0])]

    writer.add_embedding(codebook_embeddings, metadata=metadata, tag="Codebook Embeddings", global_step=0)
    print("Logged codebook embeddings to TensorBoard projector.")

    writer.close()


if __name__ == "__main__":
    main()
