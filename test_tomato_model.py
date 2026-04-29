import argparse
import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

def main(image_path):
    print(f"Loading image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("Error: Could not read image.")
        return

    # Load metadata
    meta_path = Path("outputs/tomato_model/tomato_model_meta.json")
    if not meta_path.exists():
        print("Error: Metadata file not found. Have you trained the model yet?")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    class_names = meta.get("class_names", [])
    image_size = meta.get("image_size", 128)

    # Load model
    model_path = Path("outputs/tomato_model/tomato_model.keras")
    if not model_path.exists():
        print("Error: Model file not found. Have you trained the model yet?")
        return
        
    print("Loading Deep Learning Model (Tomato Classifier)...")
    dl_model = tf.keras.models.load_model(model_path)

    # Preprocess image
    # Note: the training script scales pixels to [0, 1] instead of using keras preprocess_input
    # and resizes to image_size (default 128)
    img_resized = cv2.resize(image_bgr, (image_size, image_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    img_array = img_array.astype(np.float32) / 255.0

    # Predict
    print("\n--- Tomato Model Predictions ---")
    predictions = dl_model.predict(img_array, verbose=0)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    
    for i, idx in enumerate(top_indices):
        label = class_names[idx].replace("Tomato___", "").replace("_", " ") if class_names else str(idx)
        conf = predictions[idx]
        marker = "⭐" if i == 0 else "  "
        print(f"  {marker} {label}: {conf:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AgriVision Tomato Model via CLI")
    parser.add_argument("image_path", help="Path to the leaf image to test")
    args = parser.parse_args()
    
    main(args.image_path)
