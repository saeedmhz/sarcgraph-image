import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.lr_scheduler import CosineAnnealingLR
from flash.core.optimizers import LARS

from classification.simclr.finetuning.core import (
    load_pretrained_backbone,
    FineTuneAugment,
    FineTunedSimCLRModel,
)
from classification.simclr.pretraining.core import (
    convert_to_syncbn,
    param_groups,
)


class FineTuningTrainer(pl.LightningModule):
    """LightningModule for fine-tuning the entire SimCLR network."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Load pretrained backbone and first-layer proj weights
        backbone, proj_head = load_pretrained_backbone(
            cfg.pretrained_ckpt_path,
            dim_h=getattr(cfg, "hidden_dim", 512),
            dim_out=getattr(cfg, "output_dim", 128),
        )
        # Convert to SyncBatchNorm for multi-GPU consistency
        backbone = convert_to_syncbn(backbone)
        proj_head = convert_to_syncbn(proj_head)

        # Full model: backbone + proj_head + classifier
        self.model = FineTunedSimCLRModel(
            backbone=backbone, proj_head=proj_head, num_classes=cfg.num_classes
        )
        self.augment = FineTuneAugment(cfg.img_size, cfg.mean, cfg.std)
        # Use CrossEntropyLoss with integer labels
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch  # labels: shape [B], dtype long
        imgs = self.augment.train(imgs)
        logits = self.model(imgs)  #  / self.cfg.temperature
        loss = self.loss_fn(logits, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = self.augment.val(imgs)
        logits = self.model(imgs)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val_accuracy", acc, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        # Define parameter groups reusing pretraining logic
        pg = param_groups(self.model, self.cfg.weight_decay)
        optimizer = LARS(
            pg,
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
            trust_coefficient=getattr(self.cfg, "trust_coefficient", 0.001),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        return [optimizer], [scheduler]
