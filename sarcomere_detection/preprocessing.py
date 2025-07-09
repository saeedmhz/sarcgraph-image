# preprocessing.py

import numpy as np
import pandas as pd
import cv2

from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline
from sklearn.cluster import DBSCAN


def load_sample_data(
    sample_id,
    raw_dir,
    dino_dir,
    simclr_dir,
    zdiscs_dir,
    mask_dir,
):
    """# noqa
    Load data for a single sample (raw image, probabilities, z-discs, contours, mask).

    Parameters
    ----------
    sample_id : int
        Identifier for the sample (used in filenames).
    raw_dir : str
        Directory containing the raw image .npy files.
    dino_dir : str
        Directory containing the DINO-based probabilities.
    simclr_dir : str
        Directory containing the SimCLR-based probabilities.
    zdiscs_dir : str
        Directory containing the CSV and contour files for z-discs.
    mask_dir : str
        Directory containing the binary mask .npy files.

    Returns
    -------
    raw : np.ndarray
        2D raw image array.
    probabilities : np.ndarray
        Combined DINO+SimCLR probabilities for each contour.
    zdiscs_df : pd.DataFrame
        DataFrame read from zdiscs CSV file.
    contours : np.ndarray
        Array of contour coordinates (list-like objects).
    cell_mask : np.ndarray (bool)
        Boolean mask indicating the cell area.
    """
    # Load raw
    raw = np.load(f"{raw_dir}/raw-{sample_id}.npy")

    # Load probabilities from two sources and average them
    dino_probs = np.load(f"{dino_dir}/probs-{sample_id}.npy")
    simclr_probs = np.load(f"{simclr_dir}/probs-{sample_id}.npy")[:, 1]
    probabilities = (dino_probs + simclr_probs) / 2.0

    # Load zdiscs CSV and contour .npy
    zdiscs_df = pd.read_csv(f"{zdiscs_dir}/zdiscs-{sample_id}.csv")
    contours = np.load(
        f"{zdiscs_dir}/contours-{sample_id}.npy", allow_pickle=True
    )

    # Load mask
    cell_mask = np.load(f"{mask_dir}/mask-{sample_id}.npy") > 0

    return raw, probabilities, zdiscs_df, contours, cell_mask


def pad_image(raw, pad, constant_values=0):
    """
    Pad the raw image by 'pad' pixels on each side with a constant value.

    Parameters
    ----------
    raw : np.ndarray
        Original 2D image array.
    pad : int
        Number of pixels to pad on each border.
    constant_values : int or float
        Fill value for the padded region.

    Returns
    -------
    raw_padded : np.ndarray
        Padded image array.
    """
    return np.pad(
        raw,
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=constant_values,
    )


def filter_contours(contours, probabilities, prob_thresh):
    """
    Filter out contours whose probabilities are below a certain threshold.

    Parameters
    ----------
    contours : np.ndarray
        Array of contour arrays. Each contour is an Nx2 array of (x, y) points.
    probabilities : np.ndarray
        Array of length N with the probability for each contour.
    prob_thresh : float
        Minimum probability to keep the contour.

    Returns
    -------
    filtered_contours : np.ndarray
        Contours whose probability >= prob_thresh.
    filtered_probs : np.ndarray
        Corresponding probabilities for the filtered contours.
    """
    keep_idx = probabilities >= prob_thresh
    filtered_contours = contours[keep_idx]
    filtered_probs = probabilities[keep_idx]
    return filtered_contours, filtered_probs


def correct_zdiscs(
    raw_padded,
    contours,
    probs,
    intensity_thresh=0.6,
    initial_points=15,
    pad=64,
):
    """# noqa
    Refine/Correct Z-disc locations

    Parameters
    ----------
    raw_padded : np.ndarray
        Padded 2D image.
    contours : np.ndarray
        Contours (each is Nx2 array). Already filtered by probability if desired.
    probs : np.ndarray
        Probabilities for each contour in 'contours'.
    intensity_thresh : float
        Threshold for considering a local optimum valid (on the filtered region).
    initial_points : int
        Number of random initial points (for local search).
    pad : int
        How many pixels were padded (used to interpret boundingRect offsets, etc.).

    Returns
    -------
    zdiscs_corrected : np.ndarray
        Array of shape (M, 2) with refined x,y coordinates for each Z-disc.
    new_probs : np.ndarray
        Refined probability values for each corrected disc.
    """
    recorded_optima = []
    new_probs = []

    for contour_id, contour in enumerate(contours):
        contour = contour.astype(np.int32)
        contour = contour[:, ::-1]  # swap x,y if needed
        p = probs[contour_id]

        x, y, w, h = cv2.boundingRect(contour)
        # Expand bounding box slightly
        x, y, w, h = x - 1, y - 1, w + 2, h + 2

        # Minimum bounding size
        if w < 5:
            e = 5 - w
            x -= e // 2
            w += e
        if h < 5:
            e = 5 - h
            y -= e // 2
            h += e

        # Make a local mask for this bounding box
        mask = np.zeros(raw_padded.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Extract the region of interest
        roi_masked = (
            mask[y : y + h, x : x + w].astype(np.float32)  # noqa
            * raw_padded[y : y + h, x : x + w]  # noqa
            / 255.0
        )

        # Smooth the region for robust local minima
        roi_masked = gaussian_filter(roi_masked, sigma=2.0, radius=10)
        # Normalize
        if np.max(roi_masked) != np.min(roi_masked):
            roi_masked = (roi_masked - np.min(roi_masked)) / (
                np.max(roi_masked) - np.min(roi_masked)
            )

        # We'll invert the region for minimization
        interpolator = RectBivariateSpline(
            np.arange(h),  # row coords
            np.arange(w),  # col coords
            -roi_masked,  # negative for minima
        )

        def interpolator_function(xx):
            return interpolator.ev(xx[0], xx[1])

        # Bounds for the local search
        bounds = [(0, h - 1), (0, w - 1)]
        init_points = np.random.uniform(
            [0, 0], [h - 1, w - 1], (initial_points, 2)
        )

        # Collect all local optima that meet the intensity threshold
        optima = set()
        for pt in init_points:
            res = minimize(
                interpolator_function, pt, bounds=bounds, method="L-BFGS-B"
            )
            val_at_min = interpolator_function(res.x)
            # Because we used negative, "res.success" means local minima found
            if res.success and val_at_min < -intensity_thresh:
                # Record the optimum in integer grid coordinates
                # multiply by 10 for sub-pixel, then offset by bounding box
                ox = int(res.x[0] * 10) + 10 * y
                oy = int(res.x[1] * 10) + 10 * x
                optima.add((ox, oy))

        # Evaluate how many distinct optima we found
        if len(optima) == 0:
            # Fallback: use the mean of the original contour
            cx, cy = 10 * np.mean(contour, axis=0)
            recorded_optima.append((cy, cx))
            new_probs.append(p)

        elif len(optima) == 1:
            # Just use the single optimum
            recorded_optima.append(optima.pop())
            new_probs.append(p)

        else:
            # Use DBSCAN to find clusters among multiple local minima
            optima_list = list(optima)
            X = np.array(optima_list)
            dbscan = DBSCAN(eps=3.0, min_samples=2)
            labels = dbscan.fit_predict(X)

            contour_probs = []
            for label in set(labels):
                if label != -1:
                    # A valid cluster
                    cluster_points = X[labels == label]
                    cluster_center = np.mean(cluster_points, axis=0)
                    recorded_optima.append(
                        (int(cluster_center[0]), int(cluster_center[1]))
                    )
                    # Evaluate the negative intensity at the center
                    local_center = cluster_center / 10 - [y, x]
                    contour_probs.append(interpolator_function(local_center))
                else:
                    # Noise points
                    noise_points = X[labels == -1]
                    for point in noise_points:
                        recorded_optima.append((point[0], point[1]))
                        local_pt = point / 10 - [y, x]
                        contour_probs.append(interpolator_function(local_pt))

            if contour_probs:
                # Scale the probabilities
                contour_probs = (
                    p * np.array(contour_probs) / np.min(contour_probs)
                )
                new_probs.extend(list(contour_probs))

    # Final corrected Z-disc locations and probabilities
    zdiscs_corrected = np.array(recorded_optima, dtype=np.float32) / 10.0
    new_probs = np.array(new_probs, dtype=np.float32)

    # Filter out discs whose new probability is still below the threshold
    keep_idx = new_probs > 0.2  # could pass prob_thresh as a param if desired
    zdiscs_corrected = zdiscs_corrected[keep_idx]
    new_probs = new_probs[keep_idx]

    return zdiscs_corrected, new_probs
