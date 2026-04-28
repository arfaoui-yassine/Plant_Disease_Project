# 🌿 Plant Disease Detection & Classification System

Welcome to the **AgriVision** Plant Disease Detection System!

This project implements an intelligent, automated pipeline for detecting and classifying plant diseases from pictures of leaves. It acts as an educational and functional bridge between two different philosophies of Artificial Intelligence:
1. **Classical Machine Learning (ML)** (Handcrafted features, Support Vector Machines, Random Forests)
2. **Deep Learning (DL)** (Convolutional Neural Networks via MobileNetV2)

The system automatically downloads the massive **PlantVillage dataset** using **TensorFlow Datasets (TFDS)**, trains multiple AI models, and presents everything in an interactive **Streamlit Web Dashboard**. Additionally, it implements **Explainable AI (Grad-CAM)** so humans can physically *see* how the Deep Learning model makes its decisions.

---

## 1. 🏗️ How the Architecture Works in Detail

The code is strictly modularized into a production-ready Machine Learning Systems design.

Here is the exact data flow:

```text
Raw Image Stream (from TFDS)
        ↓
Preprocessing (Resize, Denoise, Color Space Conversion) -> [src/imagerie/preprocessing.py]
        ↓
Segmentation (HSV Thresholding, Canny/Sobel Edges, KMeans) -> [src/imagerie/segmentation.py]
        ↓
Feature Extraction (Color Histograms, GLCM Textures) -> [src/imagerie/features.py]
        ↓
Classification (Classical ML or Deep Learning) -> [src/imagerie/ml_pipeline.py] & [src/imagerie/dl_pipeline.py]
        ↓
Explainable AI (Grad-CAM Heatmaps) -> [src/imagerie/xai.py]
        ↓
Frontend Web Dashboard -> [app.py]
```

---

## 2. 🧩 Detailed Code Breakdown

### A. Data Ingestion (`src/imagerie/tfds_pipeline.py`)
Previously, this project required manually downloading gigabytes of `.zip` files. Now, **`load_tfds_pipeline()`** connects directly to the cloud via **TensorFlow Datasets (`tfds.load("plant_village")`)**.
* It downloads 54,303 images representing 38 different plant conditions.
* It parses them into a highly optimized, dynamically-batched `tf.data.Dataset` binary stream (`.tfrecord`). 
* *What it does:* Prevents your computer's RAM from crashing by streaming images in small batches during training.

### B. Preprocessing & Segmentation (`src/imagerie/preprocessing.py` & `src/imagerie/segmentation.py`)
Before classical algorithms can understand an image, it must be simplified.
* **`preprocess_image()`**: Converts the standard RGB leaf image into Grayscale and HSV (Hue, Saturation, Value) color spaces while applying Gaussian blur to remove camera noise.
* **`segment_leaf_hsv()`**: Uses HSV color thresholding to mathematically "cut" the green/yellow leaf away from the background.
* **`detect_edges()`**: Runs Sobel and Canny algorithms to find the physical, harsh outlines of the diseased spots on the leaf.

### C. Pattern Extraction (`src/imagerie/features.py`)
Classical ML models only understand spreadsheets of numbers (tabular data). 
* **`extract_feature_vector()`**: Converts the segmented leaf into a 1D array of numbers by calculating:
  * **Texture (GLCM):** Measures how rough or smooth the leaf is (e.g., powdery mildew causes rough textures).
  * **Color Histograms:** Measures exact ratios of green vs. brown vs. yellow pixels.

### D. Model Training (`run_pipeline.py`, `src/imagerie/ml_pipeline.py`, & `src/imagerie/dl_pipeline.py`)
This is the backend CLI script that orchestrates the training. 

1. **Classical ML (`ml_pipeline.py`):** 
   * Triggers **`train_evaluate_ml()`**.
   * Feeds the extracted 1D feature arrays into a **Support Vector Machine (SVM)** and a **Random Forest**. 
   * It evaluates which one is more accurate, generates a Confusion Matrix (`outputs/notebook_run/confusion_matrix_ml.png`), and saves the smartest algorithm to your disk as **`best_ml_model.joblib`**.

2. **Deep Learning (`dl_pipeline.py`):**
   * Triggers **`train_evaluate_dl()`**.
   * It skips the manual edge/color extraction entirely. It passes the raw batch stream of 224x224 leaf images into **MobileNetV2** (a pre-trained Convolutional Neural Network).
   * Over 8 epochs (passes of the dataset), the neural connections adapt to recognize the 38 diseases intrinsically. The "brain" is saved as **`outputs/notebook_run/dl/dl_model.keras`**.

### E. Explainable AI (`src/imagerie/xai.py`)
Neural networks are infamous "Black Boxes"—it is hard to know *why* they predicted a specific disease.
* **`get_gradcam_heatmap()`**: Looks at the final computational layer of MobileNetV2 and calculates the mathematical gradients flowing back through the network. It outputs a color-coded "Heatmap". Red means the AI paid heavy attention to that exact pixel; Blue means it ignored it.

---

## 3. 🖥️ How the UI acts and what it is doing right now (`app.py`)

The Frontend is built entirely in Python using **Streamlit**. It acts as the bridge preventing users from needing to run terminal commands.

When you run `streamlit run app.py`, the UI does the following:

1. **Dynamic Loading (`load_dataset_info()`):** It automatically checks your `outputs/` folder cache. Instead of looking for local `.jpg` files, it pulls the exact disease class names that your ML model learned when it trained on the TFDS dataset.
2. **Single Image Analysis Tab:** 
   * You upload a picture of a leaf. 
   * If you select **Classical ML**, it imports `preprocessing.py`, applies the HSV masks, generates the GLCM text features on the fly, and queries `best_ml_model.joblib` for a prediction.
   * If you select **Deep Learning**, it reshapes your image to 224x224, queries `dl_model.keras`, and dynamically invokes `xai.py` to overlay the glowing Grad-CAM heatmap on top of your uploaded picture.
3. **Dataset Explorer Tab:** Because TFDS abstracts away local images, the explorer now acts as a dynamic state lookup, listing the active classes the model is currently trained capable of diagnosing.

---

## 4. 🚀 How to Run the Code

### Step 1: Install Dependencies
Ensure you are using a Python virtual environment:
```bash
pip install -r requirements.txt
pip install tensorflow-datasets importlib_resources protobuf==6.31.1 scikit-image
```

### Step 2: Train the Models (Backend)
Run the pipeline generation script. *(Note: On the first run, TensorFlow Datasets will take ~10 minutes to download and compile the 54,000 images from the internet into a local database. Subsequent runs take 2 seconds).*

```bash
python run_pipeline.py --max-samples 1000 --output outputs/notebook_run --run-dl --dl-epochs 8
```
*(This will train the ML models on 1,000 samples to save RAM, and train the Deep Learning model on the full TFDS stream).*

### Step 3: Launch the Dashboard (Frontend)
Once training finishes and the models are saved to your `outputs/` folder, start the interactive web UI:

```bash
streamlit run app.py
```
Open the provided `localhost` URL in your web browser to upload images and test the models!

*(Alternatively, you can interact with the python steps identically cell-by-cell inside `notebooks/plant_disease_pipeline.ipynb`)*.