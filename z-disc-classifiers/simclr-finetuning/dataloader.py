import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning import LightningDataModule
from torchvision.transforms.functional import to_tensor

from model import FineTuneAugment


class FineTuneDataModule(LightningDataModule):
    """
    Data module for fine-tuning SimCLR. Handles data loading, splitting, and
    dynamic augmentations during training and validation.
    """

    def __init__(self, image_path, label_path, batch_size=512, num_workers=8, img_size=128, expand_data=False):
        """
        Args:
            image_path (str): Path to the images dataset (NumPy file).
            label_path (str): Path to the labels dataset (NumPy file).
            batch_size (int): Batch size for training and validation.
            num_workers (int): Number of workers for data loading.
            img_size (int): Target image size for augmentation.
            expand_data (bool): Whether to expand data with static transformations.
        """
        super().__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.expand_data = expand_data
        self.augment = FineTuneAugment(img_size, mean=[0.2470, 12.3338, 24.3036], std=[5.3444, 35.5931, 36.6319])

    def setup(self, stage=None):
        """
        Load the dataset and perform train-validation split.
        """
        # Load images and labels
        images = np.load(self.image_path)
        labels = np.load(self.label_path)

        # Convert images to tensors
        images = torch.tensor(images, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Shuffle and split the dataset
        dataset_size = len(images)
        indices = np.random.permutation(dataset_size)
        split_idx = int(0.8 * dataset_size)
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]

        train_images, train_labels = images[train_indices], labels[train_indices]
        val_images, val_labels = images[val_indices], labels[val_indices]

        # Expand training data if enabled
        if self.expand_data:
            train_images, train_labels = self.expand_dataset(train_images, train_labels)

        # Convert labels to one-hot encoding if needed
        train_labels = self.one_hot_labels(train_labels)
        val_labels = self.one_hot_labels(val_labels)

        # Create datasets
        self.train_dataset = TensorDataset(train_images, train_labels)
        self.val_dataset = TensorDataset(val_images, val_labels)

    def train_dataloader(self):
        """
        DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn_tr,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_val,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def collate_fn_tr(self, batch):
        """
        Collate function for the training DataLoader. Applies augmentations dynamically.
        """
        images, labels = zip(*batch)
        images = torch.stack(images).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        images = self.augment.train(images)  # Apply training augmentations
        labels = torch.stack(labels)
        return images, labels

    def collate_fn_val(self, batch):
        """
        Collate function for the validation DataLoader. Applies normalization only.
        """
        images, labels = zip(*batch)
        images = torch.stack(images).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        images = self.augment.val(images)  # Apply validation normalization
        labels = torch.stack(labels)
        return images, labels

    def expand_dataset(self, images, labels):
        """
        Expand the dataset using static transformations (rotations, flips, etc.).
        Args:
            images (torch.Tensor): Input image tensor of shape [N, H, W, C].
            labels (torch.Tensor): Corresponding labels tensor of shape [N].
        Returns:
            (torch.Tensor, torch.Tensor): Expanded images and labels.
        """
        orig = images.numpy()
        rot90 = np.rot90(orig, axes=(1, 2), k=1)
        rot180 = np.rot90(orig, axes=(1, 2), k=2)
        rot270 = np.rot90(orig, axes=(1, 2), k=3)

        hflip = np.flip(orig, axis=2)
        hflip_rot90 = np.rot90(hflip, axes=(1, 2), k=1)
        hflip_rot180 = np.rot90(hflip, axes=(1, 2), k=2)
        hflip_rot270 = np.rot90(hflip, axes=(1, 2), k=3)

        expanded_images = np.concatenate((orig, rot90, rot180, rot270, hflip, hflip_rot90, hflip_rot180, hflip_rot270), axis=0)
        expanded_labels = np.hstack([labels.numpy()] * 8)

        return torch.tensor(expanded_images, dtype=torch.float32), torch.tensor(expanded_labels, dtype=torch.long)

    def one_hot_labels(self, labels):
        """
        Convert labels to one-hot encoding.
        """
        num_classes = len(torch.unique(labels))
        return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
