import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class ImageDataset(Dataset):
    """
    A flexible dataset for loading image data.

    By default, this class is designed to load image data from a NumPy `.npy` file.
    However, users can modify or extend this class to load image data from other
    sources, such as LMDB, HDF5, etc.

    Args:
        image_file (str): Path to the image data (e.g., a `.npy` file).
                          The array should have shape [N, H, W, C] where:
                          - N: Number of images
                          - H, W: Image dimensions
                          - C: Number of channels (e.g., 3 for RGB)
        transform (callable, optional): A function/transform to apply to each image.
    """
    def __init__(self, image_file):
        self.images = np.load(image_file)  # Default: Load images from a NumPy file

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Convert to PyTorch tensor and permute to [C, H, W]
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)


class DataModule(LightningDataModule):
    """
    A flexible DataModule for image datasets.

    This DataModule uses `GeneralImageDataset` by default, which loads images from
    a NumPy `.npy` file. Users can modify the dataset class or logic to load data
    from other sources.

    Args:
        image_file (str): Path to the image data (e.g., a `.npy` file).
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker threads for the DataLoader.
        shuffle (bool): Whether to shuffle the data during training.
        transform (callable, optional): Transformations to apply to the dataset.
    """
    def __init__(self, image_file, batch_size=256, num_workers=16):
        super().__init__()
        self.image_file = image_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None

    def setup(self, stage=None):
        """
        Initialize the dataset. This is called once at the start of training.
        """
        self.dataset = ImageDataset(image_file=self.image_file)

    def train_dataloader(self):
        """
        Return the DataLoader for training.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )