from datetime import datetime
from pathlib import Path
from utils.torch_utils import get_device, fix_seed
import argparse
from train import train


def main(args):
    fix_seed(42)
    
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
        'total_training_images': 5_120_000,
        'eval_every': 51_200,  # 1% of total_training_images (5_120_000)
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