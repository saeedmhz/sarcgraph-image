import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights

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