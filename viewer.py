from pathlib import Path
import numpy as np
import pyviewer # pip install -e .
from imgui_bundle import imgui, implot
from pyviewer import single_image_viewer as siv
from pyviewer.toolbar_viewer import ToolbarViewer
from utils.utils import get_transform
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from utils.constants import CIFAR10_DATA_ROOT
from vqvae import VQVAE
from utils.utils import preprocess_img_tensors, get_device
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



def main():

     # Load the model checkpoint
    checkpoint_path = Path("checkpoints/vqvae_cifar10/run_2025-03-30_00-49-32/model.pth")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    device = get_device()
    model = VQVAE(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()    

    transform = get_transform()

    logging.info("Loading data...")
    test_dataset = CIFAR10(CIFAR10_DATA_ROOT, False, transform, download=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    
   

        
    siv.init('Async viewer', hidden=True)

    class Test(ToolbarViewer):
        def setup_state(self):
            self.state.seed = 0
            self.state.img = None
            self.state.index = np.random.randint(len(test_dataset))
            self.state.update_requested = True  # Trigger image load on startup
            self.test_dataset = test_dataset  # Cache the dataset
            logging.info(f"Initial random index: {self.state.index}")

        def compute(self):
            if not self.state.update_requested:
                return self.state.img  # Return cached combined image

            with torch.no_grad():
                t0 = time.time()
                image, _ = self.test_dataset[self.state.index]
                test_batch = image.unsqueeze(0).to(device)  # Add batch dimension

                # Original image
                orig_array, _, _ = preprocess_img_tensors(test_batch)

                # Reconstructed image
                output = model(test_batch)
                x_recon = output["x_recon"].cpu()
                recon_array, _, _ = preprocess_img_tensors(x_recon)

                # Concatenate side-by-side (H, W * 2, C)
                combined = np.concatenate([orig_array[0], recon_array[0]], axis=1)
                self.state.img = combined

                self.state.update_requested = False  # Reset flag
                logging.info(f"Data loading + reconstruction took {time.time() - t0:.2f}s")

            return self.state.img


        def draw_toolbar(self):
            if imgui.button("Next image"):
                self.state.index = np.random.randint(len(self.test_dataset))
                self.state.update_requested = True



    _ = Test('test_viewer')
    siv.inst.close()
    print('Done')


if __name__ == "__main__":
    main()