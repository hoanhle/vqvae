import torch
import subprocess
import numpy as np
from PIL import Image
from torchvision import transforms
import random


def fix_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def export_to_netron(model, dummy_input, filename="model.onnx"):
    torch.onnx.export(model, dummy_input, filename, 
                      input_names=["input"], output_names=["output"], opset_version=11)
    subprocess.run(["netron", filename])


def preprocess_img_tensors(img_tensors):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    return imgs_array, batch_size, img_size

def save_img_tensors_as_grid(img_tensors, nrows, f):
    imgs_array, batch_size, img_size = preprocess_img_tensors(img_tensors)
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}")


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() 
                      else "cpu")

def get_transform():
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform