from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from .preprocessing import color_histogram


def _shape_features(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))
    circularity = 0.0
    if perimeter > 1e-6:
        circularity = float(4.0 * np.pi * area / (perimeter ** 2))
    return np.array([area, perimeter, circularity], dtype=np.float32)


def _texture_glcm(gray: np.ndarray) -> np.ndarray:
    quantized = (gray / 16).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2],
        levels=16,
        symmetric=True,
        normed=True,
    )

    props = []
    for name in ["contrast", "homogeneity", "energy", "correlation"]:
        val = graycoprops(glcm, name).ravel()
        props.append(val)

    return np.concatenate(props).astype(np.float32)


def extract_feature_vector(rgb: np.ndarray, hsv: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build final handcrafted feature vector for ML models."""
    rgb_hist = color_histogram(rgb, space="rgb", bins=32)
    hsv_hist = color_histogram(hsv, space="hsv", bins=32)
    texture = _texture_glcm(gray)
    shape = _shape_features(mask)

    return np.concatenate([rgb_hist, hsv_hist, texture, shape]).astype(np.float32)
