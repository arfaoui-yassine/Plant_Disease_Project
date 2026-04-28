from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from .features import extract_feature_vector
from .preprocessing import preprocess_image
from .segmentation import detect_edges, kmeans_segmentation, segment_leaf_hsv
from .visualization import save_preprocessing_panel


def build_feature_table(
    tfds_dataset: tf.data.Dataset,
    class_names: list[str],
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a TFDS dataset.
    
    Args:
        tfds_dataset: A tf.data.Dataset with batches of (images, labels)
        class_names: List of class names in order
        out_dir: Directory to save preprocessing samples
    
    Returns:
        Tuple of (feature_matrix, labels_array)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = [], []
    sample_saved = 0

    # Create label to class name mapping (TFDS uses integer indices)
    label_to_class = {i: name for i, name in enumerate(class_names)}

    for batch_images, batch_labels in tqdm(tfds_dataset, desc="Feature extraction"):
        # batch_images: shape (batch_size, height, width, 3), dtype float32, values [0, 1]
        # batch_labels: shape (batch_size,), dtype int32, values 0 to num_classes-1
        
        # Unbatch and process each image
        for img_float32, label_idx in zip(batch_images.numpy(), batch_labels.numpy()):
            # Convert TFDS float32 [0, 1] -> uint8 BGR for OpenCV
            img_uint8 = (img_float32 * 255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            # Get class name
            class_name = label_to_class[int(label_idx)]

            pp = preprocess_image(img_bgr)
            mask = segment_leaf_hsv(pp["hsv"])
            edges = detect_edges(pp["gray"])
            km = kmeans_segmentation(pp["rgb"])

            feat = extract_feature_vector(pp["rgb"], pp["hsv"], pp["gray"], mask)
            X.append(feat)
            y.append(class_name)

            if sample_saved < 8:
                save_preprocessing_panel(
                    out_dir / "preprocessing_samples" / f"sample_{sample_saved:02d}_{class_name}.png",
                    pp["rgb"],
                    pp["gray"],
                    mask,
                    edges["sobel"],
                    edges["canny"],
                    km,
                )
                sample_saved += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y


def train_evaluate_ml(X: np.ndarray, y: np.ndarray, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc,
    )

    models = {
        "svm_rbf": Pipeline(
            steps=[("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=5, gamma="scale"))]
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }

    results = []
    best_name, best_model, best_acc = None, None, -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            pred,
            average="weighted",
            zero_division=0,
        )

        results.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    res_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    res_df.to_csv(out_dir / "ml_metrics.csv", index=False)

    y_pred_best = best_model.predict(X_test)
    labels = np.arange(len(le.classes_))
    cm = confusion_matrix(y_test, y_pred_best, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha="right")
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({best_name})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_ml.png", dpi=150)
    plt.close(fig)

    report = classification_report(
        y_test, y_pred_best, labels=labels, target_names=le.classes_, zero_division=0
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    joblib.dump({"model": best_model, "label_encoder": le}, out_dir / "best_ml_model.joblib")

    return {
        "best_model": best_name,
        "metrics": res_df.to_dict(orient="records"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
