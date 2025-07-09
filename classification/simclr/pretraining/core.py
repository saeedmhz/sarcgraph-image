import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s


def convert_to_syncbn(model):
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


def param_groups(model, wd, name="lars"):
    def skip(n):
        return "bn" in n or (name == "lars" and "bias" in n)

    g1 = {
        "params": [p for n, p in model.named_parameters() if not skip(n)],
        "weight_decay": wd,
        "layer_adaptation": True,
    }
    g2 = {
        "params": [p for n, p in model.named_parameters() if skip(n)],
        "weight_decay": 0.0,
        "layer_adaptation": False,
    }
    return [g1, g2]


class Augment:
    """
    Stochastic augmentation for SimCLR: two correlated views.
    """

    def __init__(self, img_size, mean, std, s=1.0):
        blur = T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0 * s))
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=img_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)
                ),
                T.RandomApply([blur], p=0.5),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# For out dataset: mean=[0.2470,12.3338,24.3036], std=[5.3444,35.5931,36.6319]


class ContrastiveLoss(nn.Module):
    """NT-Xent (InfoNCE) contrastive loss with dynamic mask."""

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, v1, v2):
        b = v1.size(0)
        # dynamic mask for 2B x 2B, excluding self-similarity
        mask = (~torch.eye(2 * b, dtype=torch.bool, device=v1.device)).float()
        v1n = F.normalize(v1, dim=1)
        v2n = F.normalize(v2, dim=1)
        emb = torch.cat([v1n, v2n], dim=0)
        sim = torch.mm(emb, emb.T)
        sim = sim.masked_fill(~mask.bool(), float("-inf"))
        pos = torch.cat([sim.diag(b), sim.diag(-b)], dim=0)
        num = torch.exp(pos / self.temperature)
        den = torch.sum(torch.exp(sim / self.temperature), dim=1)
        return -torch.log(num / den).mean()


class ProjectionHead(nn.Module):
    """MLP projection head."""

    def __init__(self, dim_in=1280, dim_h=512, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.BatchNorm1d(dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_h),
            nn.BatchNorm1d(dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_out),
        )

    def forward(self, x):
        return self.head(x)


class SimCLRv2Model(nn.Module):
    """EfficientNetV2-S encoder + projection head."""

    def __init__(self, dim_h=512, dim_out=128):
        super().__init__()
        self.encoder = efficientnet_v2_s(weights=None)
        self.encoder.classifier = nn.Identity()
        self.proj = ProjectionHead(1280, dim_h, dim_out)

    def forward(self, x):
        f = self.encoder(x).contiguous()
        return self.proj(f)
