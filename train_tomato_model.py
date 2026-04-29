"""
=============================================================================
  AgriVision — Tomato Disease Classifier (Standalone)
  ---------------------------------------------------
  A focused, high-performance deep learning model trained exclusively
  on tomato-related classes from the PlantVillage TFDS dataset.

  Usage:
      python train_tomato_model.py
      python train_tomato_model.py --epochs 10
=============================================================================
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


# ── 1. Configuration ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Tomato Disease Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (default: 128)")
    parser.add_argument("--output-dir", type=str, default="outputs/tomato_model", help="Output directory")
    return parser.parse_args()


# ── 2. Dataset Loading & Filtering ──────────────────────────────────────────

def load_and_filter_tomato_dataset(image_size, batch_size):
    """
    Load PlantVillage from TFDS and filter to only tomato-related classes.
    Returns train_ds, val_ds, tomato_class_names, and a label mapping.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Loading & Filtering PlantVillage Dataset")
    print("=" * 60)

    # Load full dataset with metadata
    full_ds, info = tfds.load(
        "plant_village",
        split="train",
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )

    all_class_names = info.features["label"].names
    total_classes = len(all_class_names)
    print(f"  Total classes in PlantVillage: {total_classes}")

    # ── Automatically find tomato-related classes ──
    tomato_indices = []
    tomato_class_names = []
    for idx, name in enumerate(all_class_names):
        if "Tomato" in name:
            tomato_indices.append(idx)
            tomato_class_names.append(name)

    print(f"  Detected {len(tomato_class_names)} tomato classes:")
    for i, name in enumerate(tomato_class_names):
        print(f"    [{i}] {name}")

    # Build a lookup table: old_label -> new_label (0-indexed within tomato classes)
    old_to_new = {}
    for new_idx, old_idx in enumerate(tomato_indices):
        old_to_new[old_idx] = new_idx

    num_tomato_classes = len(tomato_class_names)

    # ── Filter + remap labels ──
    tomato_index_set = set(tomato_indices)

    def is_tomato(image, label):
        """Filter predicate: keep only tomato labels."""
        return tf.reduce_any(tf.equal(label, list(tomato_index_set)))

    # Build the remap lookup as a TF constant
    remap_table = np.full(total_classes, -1, dtype=np.int32)
    for old_idx, new_idx in old_to_new.items():
        remap_table[old_idx] = new_idx
    remap_tensor = tf.constant(remap_table, dtype=tf.int32)

    def preprocess(image, label):
        """Resize, normalize, and remap label."""
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        new_label = tf.gather(remap_tensor, label)
        return image, new_label

    # Apply filter and preprocessing
    tomato_ds = full_ds.filter(is_tomato).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # ── Count total tomato samples ──
    print("\n  Counting tomato samples...")
    count = 0
    for _ in tomato_ds:
        count += 1
    print(f"  Total tomato samples: {count}")

    # ── 80/20 Train/Val Split ──
    tomato_ds = tomato_ds.shuffle(5000, seed=42)
    val_size = int(0.2 * count)
    train_size = count - val_size

    val_ds = tomato_ds.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = tomato_ds.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"  Train samples: {train_size}")
    print(f"  Val samples:   {val_size}")

    return train_ds, val_ds, tomato_class_names, num_tomato_classes


# ── 3. Model Construction ──────────────────────────────────────────────────

def build_tomato_model(image_size, num_classes):
    """Build a MobileNetV2-based classifier for tomato diseases."""
    print("\n" + "=" * 60)
    print("  PHASE 2: Building MobileNetV2 Model")
    print("=" * 60)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Freeze base initially

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ── 4. Training ────────────────────────────────────────────────────────────

def train_model(model, train_ds, val_ds, epochs):
    """Train the model and return the history."""
    print("\n" + "=" * 60)
    print(f"  PHASE 3: Training for {epochs} epoch(s)")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
    )
    return history


# ── 5. Evaluation ──────────────────────────────────────────────────────────

def evaluate_model(model, val_ds, class_names, out_dir):
    """Generate classification report, confusion matrix, and metrics."""
    print("\n" + "=" * 60)
    print("  PHASE 4: Evaluation & Metrics")
    print("=" * 60)

    # Collect all predictions and true labels
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Final accuracy & loss
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n  Final Validation Accuracy: {val_acc:.4f}")
    print(f"  Final Validation Loss:     {val_loss:.4f}")

    # Classification Report
    clean_names = [n.replace("Tomato___", "").replace("_", " ") for n in class_names]
    report = classification_report(y_true, y_pred, target_names=clean_names, zero_division=0)
    print("\n  Classification Report:")
    print("-" * 60)
    print(report)

    # Save report
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(clean_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(clean_names, fontsize=8)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Tomato Disease Classifier — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to: {out_dir / 'confusion_matrix.png'}")

    return y_true, y_pred


# ── 6. Visualizations ──────────────────────────────────────────────────────

def plot_training_history(history, out_dir):
    """Plot accuracy and loss curves."""
    print("\n" + "=" * 60)
    print("  PHASE 5: Training Visualizations")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax1.set_title("Accuracy Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax2.set_title("Loss Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Tomato Disease Classifier — Training History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png", dpi=150)
    plt.close(fig)
    print(f"  Training history plot saved to: {out_dir / 'training_history.png'}")


def plot_sample_predictions(model, val_ds, class_names, out_dir, num_samples=9):
    """Display a grid of sample predictions vs ground truth."""
    clean_names = [n.replace("Tomato___", "").replace("_", " ") for n in class_names]

    # Grab one batch
    for images, labels in val_ds.take(1):
        preds = model.predict(images, verbose=0)
        pred_indices = np.argmax(preds, axis=1)
        break

    num_samples = min(num_samples, len(images))
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        img = images[i].numpy()
        true_label = clean_names[labels[i].numpy()]
        pred_label = clean_names[pred_indices[i]]
        confidence = preds[i][pred_indices[i]]

        ax.imshow(img)
        is_correct = labels[i].numpy() == pred_indices[i]
        color = "green" if is_correct else "red"
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})",
            fontsize=9,
            color=color,
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Tomato Disease Classifier — Sample Predictions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_predictions.png", dpi=150)
    plt.close(fig)
    print(f"  Sample predictions saved to: {out_dir / 'sample_predictions.png'}")


# ── 7. Main Pipeline ───────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("  AgriVision — Tomato Disease Classifier")
    print("#" * 60)

    # Step 1: Load and filter dataset
    train_ds, val_ds, class_names, num_classes = load_and_filter_tomato_dataset(
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    # Step 2: Build model
    model = build_tomato_model(args.image_size, num_classes)

    # Step 3: Train
    history = train_model(model, train_ds, val_ds, args.epochs)

    # Step 4: Evaluate
    evaluate_model(model, val_ds, class_names, out_dir)

    # Step 5: Visualizations
    plot_training_history(history, out_dir)
    plot_sample_predictions(model, val_ds, class_names, out_dir)

    # Step 6: Save model (separate from existing models)
    model_path = out_dir / "tomato_model.keras"
    model.save(model_path)
    print(f"\n  Model saved to: {model_path}")

    # Save class names for future use
    import json
    meta = {
        "class_names": class_names,
        "num_classes": num_classes,
        "image_size": args.image_size,
        "epochs": args.epochs,
    }
    (out_dir / "tomato_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "#" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  All outputs saved to: {out_dir.absolute()}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
