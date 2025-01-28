import os
import logging
from pytorch_lightning.loggers import TensorBoardLogger
import json

def setup_logging(save_dir, log_name="training.log"):
    """
    Sets up Python logging to log to both console and a file.
    
    Args:
        save_dir (str): Directory where the log file will be saved.
        log_name (str): Name of the log file.
        
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    log_file_path = os.path.join(save_dir, log_name)
    os.makedirs(save_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),  # Log to file
            logging.StreamHandler()             # Log to console
        ]
    )
    logger = logging.getLogger()
    return logger


def save_config(config, save_dir, config_name="config.json"):
    """
    Save configuration to a JSON file.
    
    Args:
        config (object): Configuration object or dictionary.
        save_dir (str): Directory where the config file will be saved.
        config_name (str): Name of the config file.
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, config_name)
    with open(config_path, "w") as f:
        json.dump(config.__dict__ if hasattr(config, "__dict__") else config, f, indent=4)


def log_model_summary(model, input_size, save_dir, summary_name="model_summary.txt"):
    """
    Log and save the model architecture summary.
    
    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        input_size (tuple): Input size for the model summary.
        save_dir (str): Directory where the summary will be saved.
        summary_name (str): Name of the model summary file.
    """
    from torchsummary import summary
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, summary_name)
    with open(summary_path, "w") as f:
        f.write(str(summary(model, input_size)))