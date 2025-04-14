import numpy as np
import itertools

from aalto_submit import AaltoSubmission
from submit_utils import EasyDict
from utils.torch_utils import get_device, fix_seed


TRITON_GPU_ENVS = {
    'GPU-P100-16G': 16,
    'GPU-V100-16G': 16,
    'GPU-V100-32G': 32,
    'GPU-A100-80G': 80,
    'GPU-H100-80G': 80,
    'GPU-DEBUG': 16,
    'DGX-COMMON': 16,
}

RUN_DIRS = {
    **{k: f'<RESULTS>/graphics/leh19/results/vqvae' for k in TRITON_GPU_ENVS},
    'L': '/home/leh19/workspace/vqvae/log/cifar10',
}

# Local to machine running training
DSET_ROOTS = {
    **{k: '/scratch/cs/graphics/leh19/datasets/cifar10' for k in TRITON_GPU_ENVS},
    'L': '/home/leh19/datasets/cifar10',
}


ENV = 'GPU-V100-16G'


#----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Base Submission Configuration ---
    submit_config = EasyDict()
    submit_config.run_func = 'train.train'
    submit_config.run_dir_root = RUN_DIRS[ENV] + '/hyperparam_sweep'
    submit_config.username = 'leh19'
    submit_config.time = '0-01:00:00'  # Adjust time limit if needed per run
    submit_config.num_gpus = 1
    submit_config.num_cores = 2
    submit_config.cpu_memory = max(32, submit_config.num_gpus * 16)  # In GB.
    submit_config.env = ENV # Or 'GPU-V100-32G', 'GPU-A100-80G'
    submit_config.modules = ['vqvae-1.0']
    submit_config.run_dir_extra_ignores = ['log', 'data', '__pycache__']

    # --- Static Args (passed to train.train) ---
    use_ema = True

    print(f"Loading train dataset from {DSET_ROOTS[ENV]}")
    dataset_kwargs = {
        'data_root': DSET_ROOTS[ENV],
    }
    training_kwargs = {
        'total_training_images': 10_240_000,
        'eval_every': 102_400,  # 1% of total_training_images (10_240_000)
    }

    num_hiddens_values = [128, 256]
    num_residual_hiddens_values = [64, 128, 256]
    embedding_dim_values = [16, 32, 64]
    num_embeddings_values = [512, 1024, 2048]

    sweep_params = list(itertools.product(
        num_hiddens_values,
        num_residual_hiddens_values,
        embedding_dim_values,
        num_embeddings_values
    ))

    print(f"Starting hyperparameter sweep with {len(sweep_params)} combinations...")

    for i, (num_hiddens, num_residual_hiddens, embedding_dim, num_embeddings) in enumerate(sweep_params):
        print(f"Submitting run {i+1}/{len(sweep_params)}: num_hiddens={num_hiddens}, num_residual_hiddens={num_residual_hiddens}, embedding_dim={embedding_dim}, num_embeddings={num_embeddings}")

        run_func_args = EasyDict()

        run_func_args.model_kwargs = {
            "in_channels": 3,
            "num_hiddens": num_hiddens,
            "num_downsampling_layers": 2,
            "num_residual_layers": 2,
            "num_residual_hiddens": num_residual_hiddens,
            "embedding_dim": embedding_dim,
            "num_embeddings": num_embeddings,
            "use_ema": use_ema,
            "decay": 0.99,
            "epsilon": 1e-5,
        }

        # Other arguments for train.train
        run_func_args.dataset_kwargs = dataset_kwargs
        run_func_args.training_kwargs = training_kwargs
        run_func_args.device = get_device()
        # --- Configure submission details for this specific run ---
        current_submit_config = submit_config.copy()
        current_submit_config['task_description'] = f'sweep_nh{num_hiddens}_nrh{num_residual_hiddens}_ed{embedding_dim}_ne{num_embeddings}'

        submission = AaltoSubmission(run_func_args=run_func_args, **current_submit_config)

        # Run the task
        submission.run_task()
        print(f"Run {i+1} submitted.")

    print("All hyperparameter sweep jobs submitted.")