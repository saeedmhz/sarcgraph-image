import os
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import GradientAccumulationScheduler, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from augmentation import Augment
from dataloader import DataModule
from model import SimCLRTrainer
from loss import ContrastiveLoss
from logging import setup_logging, save_config, log_model_summary


class Config:
    """
    Configuration class for hyperparameters and training settings.
    """
    def __init__(self):
        self.epochs = 200  # Total training epochs
        self.seed = 42  # Random seed
        self.cuda = True  # Use GPU
        self.img_size = 128  # Input image size
        self.save_dir = "../saved_pretrained_models/"  # Save directory for checkpoints
        self.load_checkpoint = False  # Load a pretrained checkpoint if True
        self.gradient_accumulation_steps = 4  # Gradient accumulation steps
        self.batch_size = 512  # Batch size per GPU
        self.num_gpus = 2  # Number of GPUs
        self.base_lr = 0.3  # Base learning rate for a batch size of 256
        self.weight_decay = 1e-4  # Weight decay for optimizer
        self.temperature = 0.1  # Temperature for contrastive loss
        self.momentum = 0.9  # Momentum for SGD/LARS
        self.warmup_epochs = int(0.05 * self.epochs)  # Warmup epochs
        self.checkpoint_path = os.path.join(self.save_dir, "last.ckpt")  # Checkpoint path

        # Compute effective batch size
        self.effective_batch_size = self.batch_size * self.num_gpus * self.gradient_accumulation_steps

        # Dynamically scale learning rate based on effective batch size
        self.lr = self.base_lr * (self.effective_batch_size / 256)


if __name__ == "__main__":
    # Initialize configuration
    train_config = Config()
    seed_everything(train_config.seed)

    # Enable CuDNN benchmark for faster performance with consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Setup logging
    logger = setup_logging(save_dir=train_config.save_dir)
    logger.info("Starting SimCLR Pretraining...")
    
    # Save config to JSON
    save_config(train_config, train_config.save_dir)

    # Initialize augmentation pipeline
    augment = Augment(train_config.img_size)

    # Initialize the model
    model = SimCLRTrainer(
        config=train_config,
        augmentations=augment,
    )

    # Initialize the data module
    data_module = DataModule(
        image_file="path/to/dataset.npy",  # Update this to your dataset path
        batch_size=train_config.batch_size,
    )

    # Log the model architecture
    log_model_summary(
        model=model,
        input_size=(3, train_config.img_size, train_config.img_size),
        save_dir=train_config.save_dir,
    )

    # Callbacks: Gradient Accumulation and Model Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_config.save_dir,
        filename="SimCLRv2_EffNetv2_Lars_PreTrained",
        every_n_epochs=1,
        save_last=True,
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )
    accumulator = GradientAccumulationScheduler(
        scheduling={0: train_config.gradient_accumulation_steps}
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16,  # Mixed precision for faster training
        callbacks=[accumulator, checkpoint_callback],
        accelerator="gpu",
        devices=train_config.num_gpus,  # Dynamically set number of GPUs
        max_epochs=train_config.epochs,
    )

    # Train the model
    if train_config.load_checkpoint and os.path.exists(train_config.checkpoint_path):
        trainer.fit(model, data_module, ckpt_path=train_config.checkpoint_path)
    else:
        trainer.fit(model, data_module)

    logger.info("Training completed successfully.")