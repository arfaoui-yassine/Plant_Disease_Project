from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_preprocessing_panel(
    out_file: Path,
    rgb: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    sobel: np.ndarray,
    canny: np.ndarray,
    kmeans_img: np.ndarray,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Gray")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Mask HSV")

    axes[3].imshow(sobel, cmap="gray")
    axes[3].set_title("Sobel")

    axes[4].imshow(canny, cmap="gray")
    axes[4].set_title("Canny")

    axes[5].imshow(kmeans_img)
    axes[5].set_title("KMeans")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
