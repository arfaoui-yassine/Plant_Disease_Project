from __future__ import annotations

import cv2
import numpy as np


def segment_leaf_hsv(hsv_image: np.ndarray) -> np.ndarray:
    """Simple HSV segmentation for green/yellow-ish leaves.

    The threshold is intentionally broad to support multiple classes.
    """
    lower = np.array([15, 25, 20], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_edges(gray_image: np.ndarray) -> dict[str, np.ndarray]:
    """Sobel magnitude + Canny edges."""
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    canny = cv2.Canny(gray_image, 80, 160)
    return {
        "sobel": sobel_mag,
        "canny": canny,
    }


def kmeans_segmentation(rgb_image: np.ndarray, k: int = 3) -> np.ndarray:
    """Color clustering segmentation map."""
    pixels = rgb_image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(rgb_image.shape)
    return segmented
