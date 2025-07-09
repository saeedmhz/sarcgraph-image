import pytorch_lightning as pl
from flash.core.optimizers import LARS, LinearWarmupCosineAnnealingLR

from classification.simclr.pretraining.core import (
    Augment,
    ContrastiveLoss,
    SimCLRv2Model,
    convert_to_syncbn,
    param_groups,
)


# SimCLR Trainer Class
class SimCLRTrainer(pl.LightningModule):
    """LightningModule for SimCLR v2."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # automatically log all incoming parameters
        self.save_hyperparameters()
        # set defaults if missing
        hidden = getattr(cfg, "hidden_dim", 512)
        output = getattr(cfg, "output_dim", 128)
        # core model
        model = SimCLRv2Model(dim_h=hidden, dim_out=output)
        self.model = convert_to_syncbn(model)
        # augment & loss
        self.augment = Augment(cfg.img_size, cfg.mean, cfg.std)
        self.criterion = ContrastiveLoss(cfg.batch_size, cfg.temperature)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x1, x2 = self.augment(batch)
        z1, z2 = self.model(x1), self.model(x2)
        loss = self.criterion(z1, z2)
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
        # Parameter groups
        pg = param_groups(self.model, self.cfg.weight_decay)
        # Optimizer
        optimizer = LARS(
            pg,
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
            trust_coefficient=getattr(self.cfg, "trust_coefficient", 0.001),
        )
        # Print optimizer info
        eff_batch = self.cfg.batch_size * getattr(
            self.cfg, "gradient_accumulation_steps", 1
        )
        print(
            f"Using LARS optimizer | Learning Rate: {self.cfg.lr} | "
            f"Effective Batch Size: {eff_batch}"
        )
        # Scheduler
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.cfg.warmup_epochs,
            max_epochs=self.cfg.epochs,
            warmup_start_lr=getattr(self.cfg, "warmup_start_lr", 0.1),
        )
        return [optimizer], [scheduler]
