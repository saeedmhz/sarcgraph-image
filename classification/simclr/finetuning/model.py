import torch
import torch.nn as nn
import torchvision.transforms as T

from torchvision.models import efficientnet_v2_s


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, output_dim=128):
        """
        Initializes the projection head.
        :param input_dim: Dimension of the input features (from the base model).
        :param hidden_dim: Dimension of the hidden layer.
        :param output_dim: Dimension of the output features.
        """
        super(ProjectionHead, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass of the projection head.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class SimCLRv2Model(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, output_dim=128):
        """
        Initializes the SimCLRv2 model.
        :param input_dim: Input dimension from the base model.
        :param hidden_dim: Dimension of the hidden layer in the projection head.
        :param output_dim: Dimension of the output features from the projection head.
        """
        super(SimCLRv2Model, self).__init__()

        # Load the EfficientNetV2-S base model
        self.base_model = efficientnet_v2_s(weights=None)
        # self.base_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # Remove the classifier (final fully connected layer)
        self.base_model.classifier = nn.Identity()

        # Add the projection head
        self.projection_head = ProjectionHead(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the SimCLRv2 model.
        """
        features = self.base_model(x).contiguous()  # Extract features with the base model
        projected_features = self.projection_head(features).contiguous()  # Project features
        return projected_features


class FineTunedSimCLRModel(nn.Module):
    """
    Fine-tuned SimCLR model using a pretrained base and a modified projection head.
    """
    def __init__(self, base_model, projection_head_layer1, num_classes=2):
        """
        Args:
            base_model (nn.Module): Pretrained base model (EfficientNetV2-S).
            projection_head_layer1 (nn.Sequential): First layer of the projection head from pretraining.
            num_classes (int): Number of classes for classification.
        """
        super(FineTunedSimCLRModel, self).__init__()
        self.base_model = base_model  # Reuse the pretrained base model
        self.projection_head = projection_head_layer1  # Reuse the first layer of the projection head
        self.classifier = nn.Linear(512, num_classes)  # Add the final classification layer

    def forward(self, x):
        """
        Forward pass through the fine-tuned SimCLR model.
        """
        features = self.base_model(x)  # Extract features from the base model
        projected = self.projection_head(features)  # Pass through the reused projection head layer
        logits = self.classifier(projected)  # Classify the projected features
        return logits


class FineTuneAugment:
    """
    Augmentation module for fine-tuning SimCLR.
    
    - Training: Random resized crop and horizontal flip for regularization.
    - Validation: Only normalization for consistent evaluation.
    """

    def __init__(self, img_size, mean, std):
        """
        Args:
            img_size (int): Target size for the image (e.g., 128x128).
            mean (list): Normalization mean for each channel.
            std (list): Normalization standard deviation for each channel.
        """
        # Training augmentations
        self.train_transform = T.Compose([
            T.RandomResizedCrop(size=img_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(mean=mean, std=std),
        ])

        # Validation augmentations (only normalization)
        self.val_transform = T.Compose([
            T.Normalize(mean=mean, std=std),
        ])

    def train(self, x):
        """
        Apply training augmentations.
        
        Args:
            x (torch.Tensor): Input image batch.

        Returns:
            torch.Tensor: Augmented image batch.
        """
        return self.train_transform(x)

    def val(self, x):
        """
        Apply validation augmentations (normalization only).
        
        Args:
            x (torch.Tensor): Input image batch.

        Returns:
            torch.Tensor: Normalized image batch.
        """
        return self.val_transform(x)
