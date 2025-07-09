import numpy as np
import cv2


def generate_contour_images(
    segmented_zdiscs, raw_cell_image, contours, crop_size=128
):
    width = crop_size
    pad = crop_size // 2
    raw_padded = np.pad(
        raw_cell_image, ((pad, pad), (pad, pad)), mode="constant"
    )

    mask = np.zeros_like(raw_padded, dtype=np.int32)
    for i, contour in enumerate(contours):
        contour = contour[:, [1, 0]].astype(np.int32) + pad
        cv2.drawContours(
            mask, [contour], -1, color=(i + 1), thickness=cv2.FILLED
        )

    cropped_contours = []
    for mask_id in range(1, mask.max() + 1):
        x, y = segmented_zdiscs[["x", "y"]].values[mask_id - 1]
        x = int(x) + pad - width // 2
        y = int(y) + pad - width // 2

        channel_1 = np.zeros_like(raw_padded)
        channel_2 = np.zeros_like(raw_padded)
        channel_3 = raw_padded.copy()

        channel_1[mask == mask_id] = raw_padded[mask == mask_id]
        channel_2[np.logical_and(mask > 0, mask != mask_id)] = raw_padded[
            np.logical_and(mask > 0, mask != mask_id)
        ]

        crop = np.stack([channel_1, channel_2, channel_3], axis=-1)[
            x : x + width, y : y + width  # noqa: E203
        ]
        cropped_contours.append(crop)

    return np.stack(cropped_contours, axis=0)
