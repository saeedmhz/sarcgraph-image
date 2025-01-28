import torch
import pytorch_lightning as pl

from flash.core.optimizers import LARS
from pytorch.optim.lr_scheduler import LinearWarmupCosineAnnealingLR

from model import SimCLRv2Model

# Convert BatchNorm to SyncBatchNorm
def convert_model_to_syncbn(model):
    """
    Convert all BatchNorm layers in the model to SyncBatchNorm layers
    for distributed training.
    """
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


# Define Parameter Groups for Optimizer
def define_param_groups(model, weight_decay, optimizer_name):
    """
    Define parameter groups for optimizers, excluding specific layers
    (e.g., BatchNorm and biases) from weight decay and LARS adaptation.
    """
    def exclude_from_wd_and_adaptation(name):
        if "bn" in name:  # Exclude BatchNorm layers
            return True
        if optimizer_name == "lars" and "bias" in name:  # Exclude biases for LARS
            return True

    param_groups = [
        {
            "params": [
                p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)
            ],
            "weight_decay": weight_decay,
            "layer_adaptation": True,
        },
        {
            "params": [
                p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)
            ],
            "weight_decay": 0.0,
            "layer_adaptation": False,
        },
    ]
    return param_groups


# SimCLR Trainer Class
class SimCLRTrainer(pl.LightningModule):
    def __init__(
        self,
        config,
        augmentations,
    ):
        """
        PyTorch Lightning Module for SimCLR v2 training.

        Args:
            config (Config): Configuration object with training parameters.
            augmentations (object): Augmentation pipeline for training.
            input_dim (int): Input dimension for the base model.
            hidden_dim (int): Hidden dimension in the projection head.
            output_dim (int): Output dimension of the projection head.
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Initialize the SimCLR model with EfficientNet and Projection Head
        self.model = SimCLRv2Model()
        self.model = convert_model_to_syncbn(self.model)  # SyncBatchNorm conversion

        # Augmentation pipeline
        self.augmentations = augmentations

        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(
            batch_size=self.config.batch_size,
            temperature=self.config.temperature,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for SimCLR.
        """
        images = batch

        # Generate augmented views
        view_1, view_2 = self.augmentations.train(images)

        # Forward pass
        z1 = self.model(view_1)
        z2 = self.model(view_2)

        # Compute contrastive loss
        loss = self.contrastive_loss(z1, z2)

        # Log training loss
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

    def configure_optimizers(self):
        """
        Configure the optimizer and scheduler.
        """
        # Define parameter groups
        param_groups = define_param_groups(
            self.model, weight_decay=self.config.weight_decay, optimizer_name="lars"
        )

        # Initialize LARS optimizer
        optimizer = LARS(
            param_groups,
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            trust_coefficient=0.001,
        )

        print(
            f"Using LARS optimizer | "
            f"Learning Rate: {self.config.lr} | "
            f"Effective Batch Size: {self.config.batch_size * self.config.gradient_accumulation_steps}"
        )

        # Scheduler
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.config.warmup_epochs,
            max_epochs=self.config.epochs,
            warmup_start_lr=0.1,
        )

        return [optimizer], [scheduler]