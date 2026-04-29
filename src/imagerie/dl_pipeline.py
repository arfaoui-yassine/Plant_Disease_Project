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
    
    # 1. Limit dataset to 20,000 samples for speed/alignment
    full_ds = tfds_dataset.unbatch().take(20000)
    dataset_size = 20000
    val_size = int(0.2 * dataset_size)
    
    def to_categorical(image, label):
        return image, tf.one_hot(label, depth=len(class_names))

    val_ds = full_ds.take(val_size).map(to_categorical).batch(32).prefetch(tf.data.AUTOTUNE)
    train_ds = full_ds.skip(val_size).map(to_categorical).batch(32).prefetch(tf.data.AUTOTUNE)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    base._name = "mobilenetv2_1.00_224"

    inp = tf.keras.layers.Input(shape=(*image_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.models.Model(inp, out)

    print(f"\n🧠 Training on {dataset_size - val_size} samples...")
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
