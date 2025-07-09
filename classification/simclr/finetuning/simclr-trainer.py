import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.lr_scheduler import CosineAnnealingLR
from flash.core.optimizers import LARS

from model import FineTunedSimCLRModel


class FineTuningTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning a SimCLR-based model.
    """
    def __init__(self, config, model):
        """
        Args:
            config: Training configuration containing hyperparameters.
            model: Fine-tuned model initialized with a base model and new projection head.
        """
        super(FineTuningTrainer, self).__init__()
        self.config = config
        self.model = model
        self.model = self.convert_to_syncbn(self.model)  # Convert to SyncBatchNorm for DDP
        
        # Define loss function (binary classification with BCE loss)
        self.loss_fn = nn.BCEWithLogitsLoss()

    @staticmethod
    def convert_to_syncbn(model):
        """
        Converts BatchNorm layers to SyncBatchNorm for distributed training.
        """
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step: compute loss and log metrics.
        """
        images, labels = batch
        logits = self.model(images)
        logits /= self.config.temperature  # Apply temperature scaling if needed
        loss = self.loss_fn(logits, labels)
        
        # Log training loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step: compute loss and accuracy.
        """
        images, labels = batch
        logits = self.model(images)
        logits /= self.config.temperature  # Apply temperature scaling if needed
        loss = self.loss_fn(logits, labels)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        true_labels = torch.argmax(labels, dim=1)  # Assumes one-hot labels
        accuracy = (preds == true_labels).float().mean()

        # Log validation metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss, "val_accuracy": accuracy}

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        # Define parameter groups for LARS optimizer
        param_groups = self.define_param_groups(
            self.model, self.config.weight_decay, "lars"
        )

        # Use LARS optimizer
        optimizer = LARS(
            param_groups,
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        # Use CosineAnnealingLR as the learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        return [optimizer], [scheduler]

    @staticmethod
    def define_param_groups(model, weight_decay, optimizer_name):
        """
        Define parameter groups for LARS or other optimizers.
        """
        def exclude_from_wd_and_adaptation(name):
            if "bn" in name:
                return True
            if optimizer_name == "lars" and "bias" in name:
                return True

        param_groups = [
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if not exclude_from_wd_and_adaptation(name)
                ],
                "weight_decay": weight_decay,
                "layer_adaptation": True,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if exclude_from_wd_and_adaptation(name)
                ],
                "weight_decay": 0.0,
                "layer_adaptation": False,
            },
        ]
        return param_groups
