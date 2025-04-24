from pathlib import Path
import numpy as np
import argparse
from PIL import Image
from imgui_bundle import imgui, implot
from pyviewer import single_image_viewer as siv
from pyviewer.toolbar_viewer import ToolbarViewer
from utils.torch_utils import get_transform
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from utils.constants import CIFAR10_DATA_ROOT
from models.vqvae import VQVAE
from utils.torch_utils import preprocess_img_tensors, get_device
import time
import logging
logging.basicConfig(level=logging.INFO)

has_torch = False
try:
    start_time = time.time()
    import torch
    torch_import_time = time.time() - start_time
    logging.info(f"Torch import took {torch_import_time:.2f} seconds")
    import torch
    has_torch = True
except:
    pass

SAVE_DIR = Path("./saved_viewer_images")
SAVE_DIR.mkdir(exist_ok=True)


def main():
     # Load the model checkpoint
    parser = argparse.ArgumentParser(description="View VQ-VAE reconstructions and encodings.")
    parser.add_argument('--checkpoint', type=str, default="", required=True,
                        help='Path to the model checkpoint file.')
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    device = get_device()
    model = VQVAE(**checkpoint['model_kwargs']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()    

    transform = get_transform()

    logging.info("Loading data...")

    load_train = True # False for test set
    dataset = CIFAR10(CIFAR10_DATA_ROOT, load_train, transform, download=False)

    load_subset = True
    subset_size = 4096
    if load_subset: dataset = Subset(dataset, range(subset_size))
    siv.init('Async viewer', hidden=True)

    class Test(ToolbarViewer):
        def setup_state(self):
            self.state.seed = 0
            self.state.img = None
            self.state.index = np.random.randint(len(dataset))
            self.state.update_requested = True
            self.state.encoding_indices = None
            self.state.z_q = None
            self.state.heatmap_data = None
            self.test_dataset = dataset


        def compute(self):
            if not self.state.update_requested:
                return self.state.img

            with torch.no_grad():
                logging.info(f"Computing image {self.state.index}")
                t0 = time.time()

                # Load a single image
                image, _ = self.test_dataset[self.state.index]
                test_batch = image.unsqueeze(0).to(device)

                # Preprocess original image
                orig_array, _, _ = preprocess_img_tensors(test_batch)

                # Encode + Decode
                self.state.z_q = model.quantize(test_batch)
                encoding_indices = model.encoding_indices
                x_recon = model.decoder(self.state.z_q).cpu()
                recon_array, _, _ = preprocess_img_tensors(x_recon)

                # Combine original + reconstruction
                combined = np.concatenate([orig_array[0], recon_array[0]], axis=1)
                self.state.img = combined

                # Store raw encoding indices (no normalization)
                self.state.encoding_indices = (
                    encoding_indices.view(self.state.z_q.shape[2], self.state.z_q.shape[3])
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
                self.state.heatmap_data = self.state.encoding_indices.astype(np.float32)

                self.state.update_requested = False
                logging.info(f"Total compute time: {time.time() - t0:.2f}s")

            return self.state.img


        def draw_toolbar(self):
            clicked_next = imgui.button("Next image")
            if clicked_next:
                self.state.index = np.random.randint(len(self.test_dataset))
                self.state.update_requested = True

            imgui.same_line()
            clicked_save = imgui.button("Save Image")
            if clicked_save and self.state.img is not None:
                try:
                    img_to_save = (self.state.img).astype(np.uint8)
                    pil_img = Image.fromarray(img_to_save)
                    save_path = SAVE_DIR / f"image_{self.state.index}.png"
                    pil_img.save(save_path)
                    logging.info(f"Image saved to {save_path}")
                except Exception as e:
                    logging.error(f"Failed to save image: {e}")


            if hasattr(self.state, "heatmap_data") and self.state.heatmap_data is not None:
                height, width = self.state.heatmap_data.shape
                plot_size = (width * 40 * self.ui_scale, height * 40 * self.ui_scale)
                if implot.begin_plot("Encoding Indices", plot_size):
                    implot.setup_axes("Width", "Height")
                    implot.plot_heatmap(
                        "",
                        self.state.heatmap_data,
                        float(self.state.heatmap_data.min()),
                        float(self.state.heatmap_data.max()),
                        "%.0f"
                    )
                    implot.end_plot()


    _ = Test('test_viewer')
    siv.inst.close()
    print('Done')


if __name__ == "__main__":
    main()