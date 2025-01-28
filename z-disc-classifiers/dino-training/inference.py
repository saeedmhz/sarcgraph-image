import os
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset

from feature_generator import load_dino_model, transform_images, extract_features_with_dino
from model_training import MLP

def infer_images(images_path, mlp_checkpoint, output_path, batch_size=500, device='cuda'):
    """
    Perform inference on a set of images and save the probabilities.
    
    Args:
        images_path (str): Path to the numpy file containing images (N, H, W, C).
        dino_checkpoint (str): Checkpoint for the pretrained DINO model.
        mlp_checkpoint (str): Checkpoint for the trained MLP classifier.
        output_path (str): Path to save the logits or probabilities as a numpy file.
        batch_size (int): Batch size for inference.
        device (str): Device to use for inference ('cuda' or 'cpu').
    """
    # Load images
    print(f"Loading images from {images_path}...")
    images = np.load(images_path)
    print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}.")

    # Load the DINO model
    print("Loading DINO model...")
    dino_model = load_dino_model(device=device)

    # Transform images
    print("Transforming images for DINO...")
    transformed_images = transform_images(images)

    # Extract features using the DINO model
    print("Extracting feature vectors with DINO...")
    feature_vectors = extract_features_with_dino(dino_model, transformed_images, batch_size=batch_size, device=device)

    # Load the trained MLP model
    print("Loading trained MLP model...")
    input_dim = feature_vectors.shape[1]
    mlp_model = MLP(input_dim=input_dim, dropout_rate=0.5)
    mlp_model.load_state_dict(torch.load(mlp_checkpoint))
    mlp_model.to(device)
    mlp_model.eval()

    # Perform inference
    print("Performing inference with MLP...")
    dataset = TensorDataset(torch.tensor(feature_vectors, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = mlp_model(inputs)  # Raw logits
            all_logits.append(logits.cpu().numpy())

    # Concatenate logits and save to file
    all_logits = np.concatenate(all_logits, axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_logits)
    print(f"Inference completed. Logits saved to {output_path}.")

# Example usage
if __name__ == "__main__":
    images_path = "./test_images.npy"  # Path to the input images
    dino_checkpoint = None  # DINO model is loaded directly from Torch Hub
    mlp_checkpoint = "./trained_mlp_model.pth"  # Path to the trained MLP checkpoint
    output_path = "./inference_logits.npy"  # Path to save the logits

    # Run inference
    infer_images(images_path, dino_checkpoint, mlp_checkpoint, output_path)
