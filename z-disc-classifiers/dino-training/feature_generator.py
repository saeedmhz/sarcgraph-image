import numpy as np
import torch
from skimage import transform
from scipy import ndimage


# Load the DINOv2 model from Torch Hub
def load_dino_model(device='cuda'):
    """
    Load and prepare the DINOv2 model for feature extraction.
    Args:
        device (str): The device to load the model onto ('cuda' or 'cpu').
    Returns:
        torch.nn.Module: The prepared DINOv2 model.
    """
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    return model.to(device).eval()


def transform_images(images, target_size=(224, 224), sigma=2, laplacian_scaler=None):
    """
    Transform input images into a suitable format for DINO feature extraction.
    Args:
        images (numpy.ndarray): Input images as a 4D array (N, H, W, C).
        target_size (tuple): Target spatial size for resizing (height, width).
        sigma (float): Standard deviation for the Gaussian filter.
        laplacian_scaler (float or None): Scaling factor for the Laplacian channel. If None, it is calculated dynamically.
    Returns:
        numpy.ndarray: Transformed images as a 4D array (N, C, target_size[0], target_size[1]).
    """
    num_images = images.shape[0]
    transformed_images = []
    laplacian_channels = []  # Store Laplacian channels for scaler calculation

    # First pass: Compute Laplacian channels and optionally calculate the scaler
    for idx, image in enumerate(images):
        if idx % 1000 == 0:
            print(f"Processing image {idx + 1}/{num_images} (first pass)...")

        # Compute Laplacian of Gaussian-filtered channel
        channel_2 = image[:, :, 2]
        channel_2_resized = transform.resize(channel_2, target_size)
        laplacian_channel = ndimage.laplace(ndimage.gaussian_filter(channel_2_resized, sigma=sigma))
        laplacian_channels.append(laplacian_channel)

    laplacian_channels = np.array(laplacian_channels)

    # Calculate the scaler if not provided
    if laplacian_scaler is None:
        laplacian_scaler = np.std(laplacian_channels)  # Standard deviation of all Laplacian values
        print(f"Calculated Laplacian scaler (std): {laplacian_scaler:.4f}")

    # Second pass: Transform images using precomputed Laplacian channels
    for idx, (image, laplacian_channel) in enumerate(zip(images, laplacian_channels)):
        if idx % 1000 == 0:
            print(f"Preprocessing image {idx + 1}/{num_images} (second pass)...")

        # Normalize the Laplacian channel
        laplacian_channel /= laplacian_scaler

        # Generate binary versions of channels 0 and 1
        channel_0_binary = (transform.resize(image[:, :, 0] > 1, target_size)).astype(float)
        channel_1_binary = (transform.resize(image[:, :, 1] > 1, target_size)).astype(float)

        # Combine the three channels
        combined_image = np.stack([laplacian_channel, channel_0_binary, channel_1_binary], axis=0)
        transformed_images.append(combined_image)

    return np.array(transformed_images)


# Extract feature vectors using the DINO model
def extract_features_with_dino(model, images, batch_size=500, device='cuda'):
    """
    Extract feature vectors for input images using the DINOv2 model.
    Args:
        model (torch.nn.Module): The DINOv2 model for feature extraction.
        images (numpy.ndarray): Preprocessed images as a 4D array (N, C, H, W).
        batch_size (int): Number of images to process per batch.
        device (str): The device to run inference on ('cuda' or 'cpu').
    Returns:
        numpy.ndarray: Extracted feature vectors as a 2D array (N, feature_dim).
    """
    num_images = images.shape[0]
    feature_vectors = []

    with torch.no_grad():
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(start_idx + batch_size, num_images)
            print(f"Feature Vector Extraction - Images: {start_idx + 1} to {end_idx}...")

            # Prepare the batch for inference
            batch = torch.tensor(images[start_idx:end_idx], dtype=torch.float32).to(device)
            features = model(batch).cpu().numpy()
            feature_vectors.append(features.reshape(features.shape[0], -1))

    return np.concatenate(feature_vectors, axis=0)


# Main function for processing and feature extraction
def main():
    """
    Main function to preprocess images and extract features using the DINOv2 model.
    """
    # Images should be a 4D numpy array with shape (N, H, W, C), where:
    # - N: Number of images
    # - H, W: Height and width of each image (e.g., 128x128)
    # - C: Number of channels (3 channels expected, representing the target contour)
    images = ... # Example: np.load('path_to_images.npy')

    # Load the DINO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dino_model = load_dino_model(device=device)

    # Transform the images
    transformed_images = transform_images(images)

    # Extract features using the DINO model
    feature_vectors = extract_features_with_dino(dino_model, transformed_images, device=device)

    # Save the extracted features
    output_path = './dino_feature_vectors.npy'
    np.save(output_path, feature_vectors)
    print(f"Feature vectors saved to {output_path}")


if __name__ == "__main__":
    main()
