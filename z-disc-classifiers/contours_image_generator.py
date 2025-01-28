import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from sarcgraph import SarcGraph

def generate_contour_images(
    raw_cell_image,
    sg_params=None,
    zdisc_csv_path=None,
    contour_save_path=None,
    contour_image_save_path=None,
    width=128,
    remove_output_folder=True,
):
    """
    Generate contour images for a single raw cell image.

    Args:
        raw_cell_image (numpy.ndarray): Input raw cell image (2D array, np.uint8 format).
        sg_params (dict): Parameters for initializing SarcGraph.
        zdisc_csv_path (str): Path to save the z-disc CSV. If None, CSV is not saved.
        contour_save_path (str): Path to save the contour array. If None, contours are not saved.
        contour_image_save_path (str): Path to save the contour image array. If None, image is not saved.
        width (int): Width and height of the cropped contour images.
        remove_output_folder (bool): Whether to remove the "output" folder after processing.

    Returns:
        numpy.ndarray: Cropped contour images as a 4D array (N, width, width, 3).
    """
    # Ensure the input is in np.uint8 format
    if raw_cell_image.dtype != np.uint8:
        raw_cell_image = (255 * (raw_cell_image / np.max(raw_cell_image))).astype(np.uint8)

    # Set default SarcGraph parameters if not provided
    if sg_params is None:
        sg_params = {
            "zdisc_min_length": 15,
            "zdisc_max_length": 200,
            "input_type": "image",
            "sigma": 2.0,
        }

    sg = SarcGraph(**sg_params)

    # Pad the raw cell image
    pad = width // 2
    raw_padded = np.pad(
        raw_cell_image, ((pad, pad), (pad, pad)), mode="constant", constant_values=0
    )

    # Perform z-disc segmentation
    zdiscs = sg.zdisc_segmentation(raw_frames=raw_padded)
    if zdisc_csv_path:
        zdiscs.to_csv(zdisc_csv_path, index=False)

    # Load contours
    contours = np.load("output/contours.npy", allow_pickle=True)[0]
    if contour_save_path:
        np.save(contour_save_path, contours, allow_pickle=True)

    # Create a mask for contours
    mask = np.zeros_like(raw_padded, dtype=np.int32)
    for i, contour in enumerate(contours):
        contour = contour[:, [1, 0]].astype(np.int32)
        cv2.drawContours(mask, [contour], -1, color=(i + 1), thickness=cv2.FILLED)

    # Process each contour
    num_masks = mask.max()
    cropped_contours = []
    for mask_id in range(1, num_masks + 1):
        x, y = zdiscs[["x", "y"]].values[mask_id - 1]
        x = int(x) - width // 2
        y = int(y) - width // 2

        # Create three channels
        channel_1 = np.zeros_like(raw_padded)
        channel_2 = np.zeros_like(raw_padded)
        channel_3 = raw_padded.copy()

        channel_1[mask == mask_id] = raw_padded[mask == mask_id]
        channel_2[np.logical_and(mask > 0, mask != mask_id)] = raw_padded[
            np.logical_and(mask > 0, mask != mask_id)
        ]

        # Crop the contour region
        contour_cropped = np.stack([channel_1, channel_2, channel_3], axis=-1)[
            x : x + width, y : y + width
        ]
        cropped_contours.append(contour_cropped)

    cropped_contours = np.stack(cropped_contours, axis=0)

    if contour_image_save_path:
        np.save(contour_image_save_path, cropped_contours)

    # Optionally remove the "output" folder
    if remove_output_folder:
        output_folder = "output"
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            print(f"Removed output folder: {output_folder}")

    return cropped_contours


def plot_random_contours(contour_image_path, save_dir, num_samples=5):
    """
    Load contour images, randomly select and save visualizations.

    Args:
        contour_image_path (str): Path to the saved contour images (NumPy array).
        save_dir (str): Directory to save the plotted images.
        num_samples (int): Number of random samples to plot and save.

    Returns:
        None
    """
    # Load the contour images
    contours = np.load(contour_image_path)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Randomly select indices to plot
    indices = random.sample(range(contours.shape[0]), min(num_samples, contours.shape[0]))

    for idx, i in enumerate(indices):
        plt.imshow(contours[i])  # Display the RGB image
        plt.axis('off')  # Remove axes for a cleaner image
        save_path = os.path.join(save_dir, f"contour_image_{idx + 1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {save_path}")