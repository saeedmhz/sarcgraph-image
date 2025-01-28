import os
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torchvision.models import efficientnet_v2_s

from dataloader import FineTuneDataModule
from model import FineTunedSimCLRModel, SimCLRv2Model
from trainer import FineTuningTrainer
from logging import setup_logging, save_config, log_model_summary


class FineTuneConfig:
    """
    Configuration class for fine-tuning hyperparameters and settings.
    """
    def __init__(self):
        self.epochs = 100  # Fine-tuning typically requires fewer epochs
        self.seed = 42  # Random seed for reproducibility
        self.cuda = True  # Enable GPU
        self.img_size = 128  # Input image size
        self.save_dir = "../saved_finetuned_models/"  # Save directory for checkpoints
        self.load_checkpoint = True  # Load pretrained checkpoint
        self.gradient_accumulation_steps = 2  # Smaller value for gradient accumulation
        self.batch_size = 512  # Batch size per GPU
        self.num_gpus = 1  # Single GPU for fine-tuning
        self.lr = 0.005  # Learning rate for fine-tuning
        self.weight_decay = 1e-4  # Weight decay for optimizer
        self.temperature = 1.0  # Temperature for logits (may not apply for fine-tuning)
        self.momentum = 0.9  # Momentum for optimizer
        self.checkpoint_path = "../saved_pretrained_models/last.ckpt"  # Path to pretraining checkpoint


if __name__ == "__main__":
    # Initialize configuration
    fine_tune_config = FineTuneConfig()
    seed_everything(fine_tune_config.seed)

    # Enable CuDNN benchmark for faster performance with consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Setup logging
    logger = setup_logging(save_dir=fine_tune_config.save_dir)
    logger.info("Starting SimCLR Fine-Tuning...")
    
    # Save configuration to JSON
    save_config(fine_tune_config, fine_tune_config.save_dir)

    # Initialize the data module for fine-tuning
    data_module = FineTuneDataModule(
        image_path="fine-tuning-data/images.npy",  # Replace with your dataset path
        label_path="fine-tuning-data/labels.npy",  # Replace with your label path
        batch_size=fine_tune_config.batch_size,
        img_size=fine_tune_config.img_size,
    )

    # Load the pretrained SimCLR model
    pretrained_checkpoint = "path/to/pretrained_checkpoint.ckpt"
    pretrained_model = SimCLRv2Model()
    pretrained_model.load_state_dict(torch.load(pretrained_checkpoint)["state_dict"])

    # Extract the pretrained components
    base_model = pretrained_model.base_model
    projection_head_layer1 = pretrained_model.projection_head.layer1

    # Initialize the fine-tuned model
    fine_tuned_model = FineTunedSimCLRModel(
        base_model=base_model,
        projection_head_layer1=projection_head_layer1,
        num_classes=2,
    )

    # Log the model architecture
    log_model_summary(
        model=fine_tuned_model,
        input_size=(3, fine_tune_config.img_size, fine_tune_config.img_size),
        save_dir=fine_tune_config.save_dir,
    )

    # Wrap the model in FineTuningTrainer
    model = FineTuningTrainer(config=fine_tune_config, model=fine_tuned_model)

    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=fine_tune_config.save_dir,
        filename="FineTune_EffNetv2",
        monitor="val_accuracy",  # Monitor validation accuracy
        mode="max",  # Save the best validation accuracy
        save_top_k=1,  # Save only the best model
        save_last=True,  # Also save the last checkpoint
        verbose=True,  # Log checkpointing information
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16,  # Mixed precision for faster fine-tuning
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=fine_tune_config.num_gpus,  # Single GPU for fine-tuning
        max_epochs=fine_tune_config.epochs,
        log_every_n_steps=10,
    )

    # Train the model
    if fine_tune_config.load_checkpoint and os.path.exists(fine_tune_config.checkpoint_path):
        trainer.fit(model, data_module, ckpt_path=fine_tune_config.checkpoint_path)
    else:
        trainer.fit(model, data_module)

    logger.info("Fine-tuning completed successfully.")