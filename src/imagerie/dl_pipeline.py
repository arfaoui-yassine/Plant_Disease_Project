from __future__ import annotations

import importlib
from pathlib import Path


def train_evaluate_dl(
    tfds_dataset: object, # tf.data.Dataset
    class_names: list[str],
    out_dir: Path,
    image_size: tuple[int, int] = (224, 224),
    epochs: int = 8,
) -> dict:
    """Train a lightweight transfer-learning model (optional extension)."""
    try:
        tf = importlib.import_module("tensorflow")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is not available. Install tensorflow to run DL extension."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    # In TFDS 'plant_village' we only get 'train'. So we split it manually using keras helper if we wanted 
    # or just split the dataset. We'll do a simple take/skip split for validation.
    
    # Check dataset size efficiently without loading into memory
    dataset_size = tf.data.experimental.cardinality(tfds_dataset).numpy()
    
    # If cardinality is unknown (common in some TF versions/datasets), count batches
    if dataset_size < 0:
        dataset_size = 0
        for _ in tfds_dataset:
            dataset_size += 1
    
    # Since tfds_dataset is already batched, dataset_size is the number of batches.
    # Let's split by batches for simplicity.
    val_batches = max(1, int(0.2 * dataset_size))
    
    # We unbatch to apply categorical mapping and then re-batch
    # but it's better to stay batched if possible or re-batch efficiently.
    def to_categorical(image, label):
        return image, tf.one_hot(label, depth=len(class_names))

    # Split batches
    val_ds_raw = tfds_dataset.take(val_batches)
    train_ds_raw = tfds_dataset.skip(val_batches)

    train_ds = train_ds_raw.unbatch().map(to_categorical).batch(32)
    val_ds = val_ds_raw.unbatch().map(to_categorical).batch(32)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    base._name = "mobilenet_v2_base" # Explicitly name for Grad-CAM

    inp = tf.keras.layers.Input(shape=(*image_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.models.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)

    model.save(out_dir / "dl_model.keras")

    return {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "epochs": int(epochs),
        "history_keys": list(history.history.keys()),
    }
