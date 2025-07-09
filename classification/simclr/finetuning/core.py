import torch
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
)
from torchvision.models import efficientnet_v2_s

# Import ProjectionHead from pretraining core
from classification.simclr.pretraining.core import ProjectionHead


def load_pretrained_backbone(pretrained_ckpt_path, dim_h=512, dim_out=128):
    """
    Load SimCLR pretrained checkpoint and return its backbone and projection
    head with only the first layer of the projection head loaded from the
    checkpoint.
    """
    # 1) Load checkpoint
    ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    # 2) Build and load backbone
    backbone = efficientnet_v2_s(weights=None)
    backbone.classifier = nn.Identity()
    backbone_state = {
        k.replace("model.encoder.", ""): v
        for k, v in state.items()
        if k.startswith("model.encoder.")
    }
    backbone.load_state_dict(backbone_state, strict=False)

    # 3) Build a fresh projection head
    proj_head = ProjectionHead(dim_in=1280, dim_h=dim_h, dim_out=dim_out)
    # Only load weights for the first Linear + BatchNorm layers
    head_state = {}
    for k, v in state.items():
        if k.startswith("model.proj.head.0") or k.startswith(
            "model.proj.head.1"
        ):
            # strip prefix 'model.proj.' to match proj_head keys
            new_k = k.replace("model.proj.", "")
            head_state[new_k] = v
    proj_head.load_state_dict(head_state, strict=False)

    return backbone, proj_head


class FineTuneAugment:
    """Augmentations for fine-tuning: crop, flip + normalize."""

    def __init__(self, img_size, mean, std):
        self.train_t = Compose(
            [
                RandomResizedCrop(
                    img_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)
                ),
                RandomHorizontalFlip(p=0.5),
                Normalize(mean, std),
            ]
        )
        self.val_t = Compose([Normalize(mean, std)])

    def train(self, x):
        return self.train_t(x)

    def val(self, x):
        return self.val_t(x)


class FineTunedSimCLRModel(nn.Module):
    """Classifier head on pretrained SimCLR encoder+proj head."""

    def __init__(
        self, backbone: nn.Module, proj_head: nn.Module, num_classes=2
    ):
        super().__init__()
        self.backbone = backbone
        self.proj_head = proj_head
        hidden_dim = proj_head.head[-1].out_features
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.float()
        feats = self.backbone(x)
        h = self.proj_head(feats)
        return self.classifier(h)
