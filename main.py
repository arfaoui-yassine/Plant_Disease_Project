import io
import json
import base64
import cv2
import numpy as np
import joblib
import tensorflow as tf
import tensorflow_datasets as tfds
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image

from src.imagerie.preprocessing import preprocess_image
from src.imagerie.segmentation import segment_leaf_hsv, detect_edges, kmeans_segmentation
from src.imagerie.features import extract_feature_vector
from src.imagerie.xai import get_gradcam_heatmap, display_gradcam

app = FastAPI(title="AgriVision Pro API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
OUT_DIR = Path("outputs/notebook_run")
models = {"ml": None, "dl": None}

# TFDS class names (the true label order used during DL training)
# These are loaded once at startup and cached
tfds_class_names: list[str] = []

def load_models():
    global tfds_class_names
    try:
        ml_path = OUT_DIR / "best_ml_model.joblib"
        if ml_path.exists():
            models["ml"] = joblib.load(ml_path)
            print("ML Model loaded")
        
        dl_path = OUT_DIR / "dl" / "dl_model.keras"
        if dl_path.exists():
            models["dl"] = tf.keras.models.load_model(dl_path)
            print("DL Model loaded")
        
        # Load the TFDS class names in the correct order
        # This is the order the DL model was trained with
        try:
            _, ds_info = tfds.load("plant_village", split="train[:1%]", with_info=True)
            tfds_class_names = ds_info.features['label'].names
            print(f"TFDS class names loaded: {len(tfds_class_names)} classes")
        except Exception as e:
            print(f"Could not load TFDS info, falling back to run_summary.json: {e}")
            summary_path = OUT_DIR / "run_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                tfds_class_names = summary.get("selected_classes", [])
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

def image_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/api/info")
async def get_info():
    classes = []
    if models["ml"]:
        classes = list(models["ml"]["label_encoder"].classes_)
    elif (OUT_DIR / "run_summary.json").exists():
        import json
        summary = json.loads((OUT_DIR / "run_summary.json").read_text())
        classes = summary.get("selected_classes", [])
    
    return {
        "classes": classes,
        "models_loaded": {
            "ml": models["ml"] is not None,
            "dl": models["dl"] is not None
        }
    }

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 1. Preprocessing
    pp = preprocess_image(img_bgr)
    
    # 2. Segmentation
    mask = segment_leaf_hsv(pp["hsv"])
    edges = detect_edges(pp["gray"])
    km = kmeans_segmentation(pp["rgb"])
    
    # Prepare processing steps for UI
    processing_steps = {
        "original": image_to_base64(pp["rgb"]),
        "grayscale": image_to_base64(cv2.cvtColor(pp["gray"], cv2.COLOR_GRAY2BGR)),
        "hsv": image_to_base64(pp["hsv"]),
        "mask": image_to_base64(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
        "edges": image_to_base64(cv2.cvtColor(edges["canny"], cv2.COLOR_GRAY2BGR)),
        "kmeans": image_to_base64(km)
    }

    # 3. ML Prediction
    ml_result = None
    if models["ml"]:
        feat = extract_feature_vector(pp["rgb"], pp["hsv"], pp["gray"], mask)
        feat = feat.reshape(1, -1)
        pred_label = models["ml"]["model"].predict(feat)[0]
        pred_proba = models["ml"]["model"].predict_proba(feat)[0]
        class_names = models["ml"]["label_encoder"].classes_
        
        ml_result = {
            "prediction": str(pred_label),
            "confidence": float(np.max(pred_proba)),
            "probabilities": {name: float(prob) for name, prob in zip(class_names, pred_proba)}
        }

    # 4. DL Prediction
    dl_result = None
    if models["dl"]:
        img_resized = cv2.resize(img_bgr, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb.astype(np.float32), axis=0)
        # DO NOT apply preprocess_input here — it's already inside the model graph
        
        predictions = models["dl"].predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions[0])
        pred_confidence = float(predictions[0][pred_idx])
        
        # Use TFDS class names (correct order matching training labels)
        dl_classes = tfds_class_names if tfds_class_names else []
        pred_label = dl_classes[pred_idx] if pred_idx < len(dl_classes) else "Unknown"
        
        # Grad-CAM
        gradcam_b64 = None
        try:
            heatmap = get_gradcam_heatmap(models["dl"], img_array, "mobilenetv2_1.00_224", pred_index=pred_idx)
            superimposed = display_gradcam(img_rgb, heatmap, alpha=0.5)
            gradcam_b64 = image_to_base64(cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Grad-CAM error: {e}")

        dl_result = {
            "prediction": pred_label,
            "confidence": pred_confidence,
            "probabilities": {name: float(prob) for name, prob in zip(dl_classes, predictions[0])},
            "gradcam": gradcam_b64
        }

    return {
        "processing_steps": processing_steps,
        "results": {
            "ml": ml_result,
            "dl": dl_result
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
