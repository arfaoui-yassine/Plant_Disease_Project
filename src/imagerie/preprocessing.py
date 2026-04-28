from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(
    image_bgr: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    denoise: bool = True,
) -> dict[str, np.ndarray]:
    """Resize + optional denoising + color-space transforms.

    Returns a dictionary containing BGR/RGB/GRAY/HSV versions.
    """
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)

    if denoise:
        filtered = cv2.GaussianBlur(resized, (5, 5), 0)
    else:
        filtered = resized

    rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    return {
        "bgr": filtered,
        "rgb": rgb,
        "gray": gray,
        "hsv": hsv,
    }


def color_histogram(image: np.ndarray, space: str = "rgb", bins: int = 32) -> np.ndarray:
    """Compute concatenated per-channel histogram, normalized."""
    hist_list = []
    for ch in range(image.shape[2]):
        hist = cv2.calcHist([image], [ch], None, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-12)
        hist_list.append(hist)
    return np.concatenate(hist_list)
