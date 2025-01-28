import torchvision.transforms as T


class Augment:
    """
    A stochastic data augmentation module for SimCLR.

    Transforms a given data example randomly to create two correlated views
    of the same example (positive pair). Designed for domain-specific tasks
    where color jitter is not used.
    """

    def __init__(self, img_size, s=1.0):
        """
        Initialize the augmentation pipeline.

        Args:
            img_size (int): Target size of the image for cropping.
            s (float): Scaling factor for Gaussian blur strength.
        """
        # Define Gaussian blur augmentation
        blur = T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0 * s))

        # Training augmentation pipeline
        self.train_transform = T.Compose([
            T.RandomResizedCrop(
                size=img_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)
            ),
            T.RandomApply([blur], p=0.5),  # Apply Gaussian blur with 50% probability
            T.Normalize(
                mean=[0.2470, 12.3338, 24.3036],  # Replace with dataset-specific stats
                std=[5.3444, 35.5931, 36.6319],  # Replace with dataset-specific stats
            ),
        ])

    def train(self, x):
        """
        Apply train-time augmentations.

        Args:
            x (torch.Tensor): Input image batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two augmented views of the input batch.
        """
        return self.train_transform(x), self.train_transform(x)