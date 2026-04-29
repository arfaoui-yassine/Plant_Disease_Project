import cv2
import numpy as np
import joblib
import tensorflow as tf
import json
import argparse
from pathlib import Path

from src.imagerie.preprocessing import preprocess_image
from src.imagerie.segmentation import segment_leaf_hsv
from src.imagerie.features import extract_feature_vector
from src.imagerie.xai import get_gradcam_heatmap, display_gradcam

def main(image_path, run_gradcam=False):
    print(f"Loading image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Load class names
    summary_path = Path("outputs/notebook_run/run_summary.json")
    class_names = []
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            data = json.load(f)
            class_names = data.get("selected_classes", [])

    print(f"Loaded {len(class_names)} classes.")
    print("-" * 40)

    # 1. Test Deep Learning Model
    print("Testing Deep Learning Model (MobileNetV2)...")
    dl_model_path = Path("outputs/notebook_run/dl/dl_model.keras")
    if dl_model_path.exists():
        dl_model = tf.keras.models.load_model(dl_model_path)
        
        img_resized = cv2.resize(image_bgr, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb, axis=0)
        img_array = img_array.astype(np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)


        
        predictions = dl_model.predict(img_array, verbose=0)[0]
        top_indices = predictions.argsort()[-3:][::-1]
        
        print(f"DL Predictions (Top 3):")
        for i, idx in enumerate(top_indices):
            label = class_names[idx] if class_names else str(idx)
            conf = predictions[idx]
            marker = "⭐" if i == 0 else "  "
            print(f"  {marker} {label}: {conf:.2%}")
        
        pred_idx = top_indices[0]
        confidence = predictions[pred_idx]
        pred_label = class_names[pred_idx] if class_names else str(pred_idx)

        if run_gradcam:
            try:
                print("Generating Grad-CAM heatmap...")
                heatmap = get_gradcam_heatmap(dl_model, img_array, "mobilenetv2_1.00_224", pred_index=pred_idx)
                superimposed = display_gradcam(img_rgb, heatmap, alpha=0.5)
                # Convert back to BGR for saving with OpenCV
                save_img = cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
                out_path = "dl_attention_result.png"
                cv2.imwrite(out_path, save_img)
                print(f"✅ Grad-CAM visualization saved to: {out_path}")
            except Exception as e:
                print(f"Could not generate Grad-CAM: {e}")
    else:
        print("DL Model not found.")

    print("-" * 40)

    # 2. Test Classical ML Model
    print("Testing Classical ML Model (SVM/RF)...")
    ml_model_path = Path("outputs/notebook_run/best_ml_model.joblib")
    if ml_model_path.exists():
        ml_model = joblib.load(ml_model_path)
        
        # Preprocessing & Feature Extraction
        pp = preprocess_image(image_bgr)
        mask = segment_leaf_hsv(pp["hsv"])
        feat = extract_feature_vector(pp["rgb"], pp["hsv"], pp["gray"], mask)
        feat = feat.reshape(1, -1)
        
        pred_idx = ml_model["model"].predict(feat)[0]
        if hasattr(ml_model["model"], "predict_proba"):
            pred_proba = ml_model["model"].predict_proba(feat)[0]
            confidence = np.max(pred_proba)
        else:
            confidence = 1.0 # Default if probabilities not enabled
        
        le = ml_model.get("label_encoder", None)
        if le is not None:
            pred_label = le.inverse_transform([pred_idx])[0]
        else:
            pred_label = str(pred_idx)

        
        print(f"ML Prediction: {pred_label} (Confidence: {confidence:.2%})")
    else:
        print("ML Model not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AgriVision models via CLI")
    parser.add_argument("image_path", help="Path to the leaf image to test")
    parser.add_argument("--gradcam", action="store_true", help="Generate and save Grad-CAM heatmap")

    args = parser.parse_args()
    
    main(args.image_path, args.gradcam)
