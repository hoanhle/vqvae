from torchvision.datasets import CIFAR10, ImageNet
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def get_cifar10_datasets(data_dir: str = "./data/cifar10") -> Tuple[CIFAR10, CIFAR10]:
    """Download and load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    target_path = Path(data_dir)
    transform = transforms.ToTensor()
    
    trainset = CIFAR10(root=target_path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=target_path, train=False, download=True, transform=transform)
    
    print(f"CIFAR-10 downloaded to: {target_path}")
    return trainset, testset

def get_imagenet_datasets(data_dir: str = "/home/leh19/scratch/graphics/datasets/ImageNet/ILSVRC2012") -> Tuple[ImageNet, ImageNet]:
    """Download and load ImageNet dataset.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    target_path = Path(data_dir)
    
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    trainset = ImageNet(root=target_path, split='train', transform=train_transform)
    valset = ImageNet(root=target_path, split='val', transform=val_transform)
    
    print(f"ImageNet downloaded to: {target_path}")
    return trainset, valset

def visualize_samples(
    dataset: CIFAR10,
    num_samples: int = 5,
    figsize: Tuple[int, int] = (15, 5),
    title: Optional[str] = None
) -> None:
    """Visualize sample images from a dataset.
    
    Args:
        dataset: Dataset to visualize samples from
        num_samples: Number of samples to display
        figsize: Figure size for the plot
        title: Optional title for the figure
    """
    plt.figure(figsize=figsize)
    
    for i in range(num_samples):
        img_tensor, label = dataset[i]
        
        # Convert tensor to NumPy for plotting
        img = img_tensor.permute(1, 2, 0).numpy()
        
        # Get class name if available
        if hasattr(dataset, 'classes'):
            label_name = dataset.classes[label]
        else:
            label_name = f"Class {label}"
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {label_name}")
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    # # Load CIFAR-10 dataset
    # trainset, testset = get_cifar10_datasets()
    
    # # Visualize training samples
    # visualize_samples(
    #     trainset,
    #     num_samples=5,
    #     title="CIFAR-10 Training Samples"
    # )
    
    # # Visualize test samples
    # visualize_samples(
    #     testset,
    #     num_samples=5,
    #     title="CIFAR-10 Test Samples"
    # )
    
    # Uncomment to load ImageNet dataset (requires significant disk space)
    imagenet_train, imagenet_val = get_imagenet_datasets()
    visualize_samples(
        imagenet_train,
        num_samples=5,
        title="ImageNet Training Samples"
    )

if __name__ == "__main__":
    main()


