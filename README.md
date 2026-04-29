# 🌿 AgriVision Pro — AI Plant Disease Detection System

An end-to-end, production-ready pipeline for detecting and classifying plant diseases from leaf images. Combines **Deep Learning** (MobileNetV2), **Classical Machine Learning** (Random Forest / SVM), and **Explainable AI** (Grad-CAM) into a unified diagnostic platform with a premium Streamlit dashboard.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Model Architecture](#-model-architecture)
- [Inference Pipeline](#-inference-pipeline)
- [Explainable AI (Grad-CAM)](#-explainable-ai-grad-cam)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Getting Started](#-getting-started)
- [CLI Tools](#-cli-tools)
- [Results](#-results)

---

## 🏗️ System Architecture

The system is composed of four major subsystems: **Data Ingestion**, **Training**, **Inference**, and **Visualization**. They interact as follows:

```mermaid
graph TB
    subgraph DATA["📥 Data Layer"]
        TFDS["TensorFlow Datasets<br/>(PlantVillage)"]
        STREAM["tf.data Streaming Pipeline"]
        TFDS --> STREAM
    end

    subgraph TRAIN["🧠 Training Layer"]
        direction LR
        ML_TRAIN["Classical ML Pipeline<br/>Feature Extraction → RF/SVM"]
        DL_TRAIN["Deep Learning Pipeline<br/>MobileNetV2 Transfer Learning"]
        TOM_TRAIN["Tomato Specialist<br/>MobileNetV2 (Filtered)"]
    end

    subgraph MODELS["💾 Saved Models"]
        ML_MODEL["best_ml_model.joblib"]
        DL_MODEL["dl_model.keras"]
        TOM_MODEL["tomato_model.keras"]
    end

    subgraph INFERENCE["🔬 Inference Layer"]
        APP["Streamlit Dashboard<br/>(app.py)"]
        CLI["CLI Tools<br/>(test_inference.py)"]
        GCAM["Grad-CAM XAI<br/>(xai.py)"]
    end

    STREAM --> ML_TRAIN
    STREAM --> DL_TRAIN
    STREAM --> TOM_TRAIN
    ML_TRAIN --> ML_MODEL
    DL_TRAIN --> DL_MODEL
    TOM_TRAIN --> TOM_MODEL
    ML_MODEL --> APP
    DL_MODEL --> APP
    TOM_MODEL --> APP
    ML_MODEL --> CLI
    DL_MODEL --> CLI
    DL_MODEL --> GCAM
    GCAM --> APP

    style DATA fill:#0d1b2a,stroke:#64ffda,color:#ccd6f6
    style TRAIN fill:#112240,stroke:#64ffda,color:#ccd6f6
    style MODELS fill:#1a1a2e,stroke:#ffb74d,color:#ccd6f6
    style INFERENCE fill:#0a192f,stroke:#64ffda,color:#ccd6f6
```

---

## 📂 Project Structure

```
Upgraded_Plant_Disease_Project/
│
├── app.py                          # 🖥️  Streamlit dashboard (main UI)
├── run_pipeline.py                 # 🚀 CLI orchestrator for full training
├── train_tomato_model.py           # 🍅 Standalone tomato classifier trainer
├── test_inference.py               # 🧪 CLI inference tester (all models)
├── test_tomato_model.py            # 🧪 CLI tomato model tester
├── AgriVision_Master_Pipeline.ipynb# 📓 Interactive Jupyter notebook
├── requirements.txt                # 📦 Python dependencies
│
├── src/imagerie/                   # 🔧 Core processing modules
│   ├── preprocessing.py            #    Image preprocessing (grayscale, HSV, blur)
│   ├── segmentation.py             #    Leaf segmentation (HSV mask, edges, K-means)
│   ├── features.py                 #    Feature extraction (histograms, texture)
│   ├── tfds_pipeline.py            #    TFDS data loading & streaming
│   ├── ml_pipeline.py              #    Classical ML training (RF, SVM)
│   ├── dl_pipeline.py              #    Deep Learning training (MobileNetV2)
│   ├── xai.py                      #    Grad-CAM explainability
│   └── visualization.py            #    Plotting utilities
│
├── outputs/
│   ├── notebook_run/               # Full model outputs
│   │   ├── dl/dl_model.keras       #    Trained DL model (38 classes)
│   │   ├── best_ml_model.joblib    #    Trained ML model + LabelEncoder
│   │   ├── run_summary.json        #    Training metrics & class list
│   │   └── confusion_matrix_ml.png #    ML confusion matrix
│   │
│   └── tomato_model/               # Tomato specialist outputs
│       ├── tomato_model.keras       #    Trained tomato DL model (10 classes)
│       ├── tomato_model_meta.json   #    Class names & config
│       ├── training_history.png     #    Accuracy/loss curves
│       ├── confusion_matrix.png     #    Confusion matrix heatmap
│       ├── sample_predictions.png   #    Visual prediction samples
│       └── classification_report.txt#    Precision/Recall/F1 report
│
└── venv/                           # Python virtual environment
```

---

## 🔄 Data Pipeline

All data is streamed from **TensorFlow Datasets (TFDS)** — no manual downloading required.

```mermaid
graph LR
    A["TFDS PlantVillage<br/>54,303 images"] --> B["tf.data.Dataset<br/>(streaming)"]
    B --> C{"Filter?"}
    C -->|All classes| D["Full Dataset<br/>38 classes"]
    C -->|Tomato only| E["Tomato Subset<br/>10 classes"]

    D --> F["Resize to 224×224"]
    E --> G["Resize to 128×128"]

    F --> H["Cast to float32<br/>(keep 0–255 range)"]
    G --> I["Normalize to 0–1"]

    H --> J["Batch (32)<br/>Prefetch (AUTOTUNE)"]
    I --> J

    J --> K["Train/Val Split<br/>(80/20)"]

    style A fill:#112240,stroke:#64ffda,color:#ccd6f6
    style K fill:#112240,stroke:#64ffda,color:#ccd6f6
```

> **Important**: The Full DL model receives images in `[0, 255]` range because MobileNetV2's built-in `preprocess_input` handles the normalization internally. The Tomato model uses standard `[0, 1]` normalization.

---

## 🧠 Model Architecture

### Model 1: Full DL — MobileNetV2 (38 Classes)

```mermaid
graph TD
    INPUT["Input Layer<br/>(224 × 224 × 3)"] --> PREPROCESS["MobileNetV2<br/>preprocess_input<br/>(scales to [-1, 1])"]
    PREPROCESS --> BASE["MobileNetV2 Base<br/>(ImageNet weights, frozen)<br/>Output: 7×7×1280"]
    BASE --> GAP["GlobalAveragePooling2D<br/>Output: 1280"]
    GAP --> DROP["Dropout (0.2)"]
    DROP --> DENSE["Dense (38, softmax)"]
    DENSE --> OUTPUT["Prediction<br/>(38 disease classes)"]

    style INPUT fill:#0d1b2a,stroke:#64ffda,color:#e6f1ff
    style BASE fill:#112240,stroke:#ffb74d,color:#e6f1ff
    style OUTPUT fill:#0d1b2a,stroke:#64ffda,color:#64ffda
```

### Model 2: Tomato DL — MobileNetV2 (10 Classes)

```mermaid
graph TD
    INPUT2["Input Layer<br/>(128 × 128 × 3)"] --> BASE2["MobileNetV2 Base<br/>(ImageNet weights, frozen)<br/>Output: 4×4×1280"]
    BASE2 --> GAP2["GlobalAveragePooling2D<br/>Output: 1280"]
    GAP2 --> DENSE2A["Dense (128, ReLU)"]
    DENSE2A --> DROP2["Dropout (0.3)"]
    DROP2 --> DENSE2B["Dense (10, softmax)"]
    DENSE2B --> OUTPUT2["Prediction<br/>(10 tomato diseases)"]

    style INPUT2 fill:#0d1b2a,stroke:#ef5350,color:#e6f1ff
    style BASE2 fill:#112240,stroke:#ef5350,color:#e6f1ff
    style OUTPUT2 fill:#0d1b2a,stroke:#ef5350,color:#ef5350
```

### Model 3: Classical ML Pipeline

```mermaid
graph LR
    IMG["Leaf Image<br/>(any size)"] --> PRE["Preprocessing<br/>RGB → BGR → HSV<br/>Grayscale, Blur"]
    PRE --> SEG["Segmentation<br/>HSV Mask<br/>(leaf isolation)"]
    SEG --> FEAT["Feature Extraction<br/>• H/S Histograms (32d)<br/>• Area Ratio (1d)<br/>• Texture StdDev (1d)<br/>Total: 34 features"]
    FEAT --> SCALE["StandardScaler"]
    SCALE --> CLF{"Classifier"}
    CLF --> RF["Random Forest<br/>(300 trees)"]
    CLF --> SVM["SVM RBF<br/>(C=5, gamma=scale)"]
    RF --> PRED["Prediction + Label"]
    SVM --> PRED

    style IMG fill:#0d1b2a,stroke:#ffb74d,color:#e6f1ff
    style FEAT fill:#112240,stroke:#ffb74d,color:#e6f1ff
    style PRED fill:#0d1b2a,stroke:#ffb74d,color:#ffb74d
```

---

## 🔬 Inference Pipeline

When a user uploads an image (via Streamlit or CLI), the system routes through the selected model:

```mermaid
flowchart TD
    UPLOAD["📤 User Uploads Image"] --> SELECT{"Model Selection"}

    SELECT -->|"🌿 Full DL"| DL_PATH["Resize 224×224<br/>preprocess_input()"]
    SELECT -->|"🍅 Tomato DL"| TOM_PATH["Resize 128×128<br/>Normalize /255"]
    SELECT -->|"🤖 Classical ML"| ML_PATH["BGR → HSV → Mask<br/>Extract 34 Features"]

    DL_PATH --> DL_PRED["MobileNetV2 (38 cls)<br/>predict()"]
    TOM_PATH --> TOM_PRED["MobileNetV2 (10 cls)<br/>predict()"]
    ML_PATH --> ML_PRED["RF/SVM<br/>predict()"]

    DL_PRED --> TOP3["Top-3 Predictions<br/>+ Confidence %"]
    TOM_PRED --> TOP3
    ML_PRED --> ML_OUT["Prediction<br/>+ Confidence %"]

    TOP3 --> DISPLAY["🖥️ Dashboard Display<br/>• Disease name<br/>• Confidence bar<br/>• Grad-CAM heatmap<br/>• Preprocessing views"]
    ML_OUT --> DISPLAY

    style UPLOAD fill:#0d1b2a,stroke:#64ffda,color:#e6f1ff
    style DISPLAY fill:#112240,stroke:#64ffda,color:#64ffda
```

---

## 🔥 Explainable AI (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) reveals **where** the model is looking when making a diagnosis. This builds trust by showing the neural network's reasoning.

```mermaid
graph TD
    INPUT["Input Image"] --> FWD["Forward Pass<br/>through MobileNetV2"]
    FWD --> CONV["Last Conv Layer<br/>(out_relu)<br/>7×7×1280 feature maps"]
    FWD --> PRED["Predicted Class<br/>(e.g., Bacterial Spot)"]

    PRED --> GRAD["Backprop Gradients<br/>∂prediction / ∂feature_maps"]
    GRAD --> POOL["Global Average Pool<br/>gradients → weights"]
    POOL --> WEIGHT["Weighted Sum<br/>weights × feature maps"]
    WEIGHT --> HEAT["ReLU + Normalize<br/>→ Heatmap (0–1)"]
    HEAT --> OVERLAY["Superimpose on<br/>Original Image"]
    CONV --> WEIGHT

    OVERLAY --> RESULT["🔥 Grad-CAM Output<br/>Red = High Attention<br/>Blue = Low Attention"]

    style INPUT fill:#0d1b2a,stroke:#64ffda,color:#e6f1ff
    style RESULT fill:#112240,stroke:#ef5350,color:#ef5350
```

**How it handles nested models**: The system automatically unwraps the MobileNetV2 base from inside the classifier wrapper, hooks into the `out_relu` layer for gradient computation, and reconstructs the forward pass through the remaining classification head layers.

---

## 🖥️ Streamlit Dashboard

The dashboard provides 4 pages accessible via the sidebar:

```mermaid
graph TD
    SIDEBAR["🔧 Sidebar<br/>• Navigation<br/>• Model Selector<br/>• System Status"] --> HOME["🏠 Home<br/>Metric cards, architecture overview"]
    SIDEBAR --> DIAG["🔬 Diagnose<br/>Upload → Predict → Visualize"]
    SIDEBAR --> PERF["📊 Performance<br/>Side-by-side model comparison"]
    SIDEBAR --> TOMATO["🍅 Tomato Insights<br/>Training curves, confusion matrix"]

    DIAG --> M1["🌿 Full DL (38 cls)"]
    DIAG --> M2["🍅 Tomato DL (10 cls)"]
    DIAG --> M3["🤖 Classical ML (RF/SVM)"]

    M1 --> RES["Results Display<br/>• Prediction + confidence<br/>• Top-3 alternatives<br/>• Grad-CAM tab<br/>• Preprocessing tab"]
    M2 --> RES
    M3 --> RES

    style SIDEBAR fill:#0a192f,stroke:#64ffda,color:#ccd6f6
    style DIAG fill:#112240,stroke:#64ffda,color:#ccd6f6
    style RES fill:#1a1a2e,stroke:#64ffda,color:#64ffda
```

### Model Selector

| Selection | Model File | Input Size | Normalization | Classes |
|-----------|-----------|:----------:|:-------------:|:-------:|
| 🌿 Full DL | `dl_model.keras` | 224×224 | `preprocess_input` ([-1,1]) | 38 |
| 🍅 Tomato DL | `tomato_model.keras` | 128×128 | `/255.0` ([0,1]) | 10 |
| 🤖 Classical ML | `best_ml_model.joblib` | Any | N/A (feature-based) | 38 |

---

## 🚀 Getting Started

### 1. Setup Environment

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train full pipeline (ML + DL, 38 classes)
python run_pipeline.py --run-dl --max-samples 20000 --dl-epochs 5

# Train tomato specialist (10 classes)
python train_tomato_model.py --epochs 5
```

### 3. Launch Dashboard

```bash
streamlit run app.py
```

---

## 🧪 CLI Tools

### Test All Models on an Image

```bash
python test_inference.py "path/to/leaf.jpg"
# Shows Top-3 DL predictions + ML prediction

python test_inference.py "path/to/leaf.jpg" --gradcam
# Also saves dl_attention_result.png
```

### Test Tomato Model Only

```bash
python test_tomato_model.py "path/to/tomato_leaf.jpg"
# Shows Top-3 tomato disease predictions
```

---

## 📊 Results

### Tomato Specialist Model (10 classes)

| Metric | Score |
|--------|:-----:|
| **Validation Accuracy** | **95%** |
| Weighted Precision | 0.95 |
| Weighted Recall | 0.95 |
| Weighted F1 | 0.95 |

**Per-class highlights:**

| Disease | Precision | Recall | F1 |
|---------|:---------:|:------:|:--:|
| Bacterial Spot | 0.98 | 0.96 | 0.97 |
| Yellow Leaf Curl Virus | 0.99 | 1.00 | 0.99 |
| Tomato Mosaic Virus | 0.97 | 0.97 | 0.97 |
| Healthy | 0.97 | 0.97 | 0.97 |
| Late Blight | 0.90 | 0.97 | 0.94 |

### Classical ML Model (38 classes)

| Model | Accuracy | F1 |
|-------|:--------:|:--:|
| Random Forest (300 trees) | 73% | 0.70 |
| SVM (RBF, C=5) | 72.5% | 0.71 |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow` | DL training, MobileNetV2, TFDS |
| `tensorflow-datasets` | PlantVillage data streaming |
| `scikit-learn` | RF, SVM, metrics, preprocessing |
| `opencv-python` | Image processing, segmentation |
| `streamlit` | Interactive web dashboard |
| `joblib` | Model serialization (ML) |
| `matplotlib` | Plotting, Grad-CAM colormap |
| `pandas` | Data manipulation |
| `tqdm` | Progress bars |

---

## 📜 License

This project is developed as part of an academic engineering curriculum for plant disease image processing and classification.