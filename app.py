import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import io

from src.imagerie.preprocessing import preprocess_image
from src.imagerie.segmentation import segment_leaf_hsv, detect_edges, kmeans_segmentation
from src.imagerie.features import extract_feature_vector
from src.imagerie.xai import get_gradcam_heatmap, display_gradcam

# Configure page
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* App Background & Base Theme */
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, hsla(145,100%,20%,0.2) 0px, transparent 50%),
            radial-gradient(at 100% 0%, hsla(210,100%,15%,0.2) 0px, transparent 50%);
        color: #f8fafc;
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #34d399, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 2rem;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #e2e8f0;
        font-weight: 600;
        border-bottom: none;
    }
    
    /* Glassmorphism Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
        border-color: rgba(52, 211, 153, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(52, 211, 153, 0.1);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #cbd5e1;
    }
    
    /* Sidebar Navigation Pills (Radio Buttons) */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.02);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(52, 211, 153, 0.08);
        border-color: rgba(52, 211, 153, 0.3);
        transform: translateX(4px);
    }
    /* Attempt to hide radio circles and center text */
    [data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 0.75rem;
        padding: 0.5rem;
        gap: 0.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #34d399;
        background: rgba(52, 211, 153, 0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.4);
        color: white;
        border: none;
    }
    
    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Alerts/Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #f8fafc;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌿 Plant Disease Detection & Classification System")
st.markdown("*Intelligent diagnosis combining Classical ML and Deep Learning with Explainable AI*")

# Sidebar for navigation
st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; margin-top: 1rem;'>
        <h1 style='color: #34d399; margin-bottom: 0; font-size: 2.2rem;'>🌿</h1>
        <h2 style='color: #f8fafc; font-weight: 800; margin-top: 0.5rem; margin-bottom: 0;'>Agri<span style='color: #34d399;'>Vision</span></h2>
        <p style='color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem;'>AI Disease Detection</p>
    </div>
""", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='font-size: 0.9rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;'>Menu</h3>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["🏠 Home", "🔍 Single Image Analysis", "📊 Model Comparison", "📈 Dataset Explorer"],
    label_visibility="collapsed"
)

# Load models and data
@st.cache_resource
def load_ml_model():
    try:
        model_path = Path("outputs/notebook_run/best_ml_model.joblib")
        if model_path.exists():
            return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Could not load ML model: {e}")
    return None

@st.cache_resource
def load_dl_model():
    try:
        model_path = Path("outputs/notebook_run/dl/dl_model.keras")
        if model_path.exists():
            return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.warning(f"Could not load DL model: {e}")
    return None

@st.cache_data
def load_dataset_info():
    # Use joblib to pull classes from the trained model automatically instead of file folders
    try:
        model_path = Path("outputs/notebook_run/best_ml_model.joblib")
        if model_path.exists():
            data = joblib.load(model_path)
            return list(data["label_encoder"].classes_)
    except Exception:
        pass
    return ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"] # Default fallback if no model trained yet

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Project Overview")
        st.markdown("""
        This system combines **classical machine learning** and **deep learning** techniques 
        to automatically detect and classify plant diseases from leaf images.
        
        **Key Features:**
        - 🖼️ Image preprocessing and segmentation
        - 🎯 Handcrafted feature extraction
        - 🤖 Multiple ML classifiers (SVM, Random Forest)
        - 🧠 Deep learning with transfer learning
        - 💡 Explainable AI (Grad-CAM) for model interpretability
        """)
    
    with col2:
        st.markdown("### 🎯 Pipeline Architecture")
        st.markdown("""
        ```
        Input Image
            ↓
        Preprocessing (Resize, Denoise, Color Conversion)
            ↓
        Segmentation (HSV, Canny, Sobel, K-means)
            ↓
        Feature Extraction (Color, Texture, Shape)
            ↓
        Classification (ML or DL)
            ↓
        Explainability (Grad-CAM for DL)
            ↓
        Prediction & Confidence
        ```
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Classes", len(load_dataset_info()))
    with col2:
        st.metric("ML Models", "2 (SVM, RF)")
    with col3:
        st.metric("DL Model", "MobileNetV2")

# ==================== SINGLE IMAGE ANALYSIS ====================
elif page == "🔍 Single Image Analysis":
    st.header("Single Image Analysis")
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Select Model Type:", ["Classical ML", "Deep Learning"])
    with col2:
        st.info("💡 Classical ML uses handcrafted features. DL uses automatic feature extraction.")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload a leaf image (JPG, PNG):", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📸 Original", "🔧 Preprocessing", "🎯 Segmentation", "🤖 Prediction"])
        
        with tab1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with tab2:
            st.subheader("Preprocessing Steps")
            pp = preprocess_image(image_bgr)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(pp["rgb"], cv2.COLOR_BGR2RGB), caption="RGB Image")
            with col2:
                st.image(pp["gray"], caption="Grayscale")
            with col3:
                st.image(cv2.cvtColor(pp["hsv"], cv2.COLOR_HSV2RGB), caption="HSV Image")
        
        with tab3:
            st.subheader("Segmentation Results")
            pp = preprocess_image(image_bgr)
            mask = segment_leaf_hsv(pp["hsv"])
            edges = detect_edges(pp["gray"])
            km = kmeans_segmentation(pp["rgb"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(mask, caption="HSV Mask")
            with col2:
                st.image(edges["canny"], caption="Canny Edges")
            with col3:
                st.image(km, caption="K-means Segmentation")
        
        with tab4:
            st.subheader("Classification Results")
            
            if model_type == "Classical ML":
                ml_model = load_ml_model()
                if ml_model is not None:
                    # Extract features
                    pp = preprocess_image(image_bgr)
                    mask = segment_leaf_hsv(pp["hsv"])
                    feat = extract_feature_vector(pp["rgb"], pp["hsv"], pp["gray"], mask)
                    feat = feat.reshape(1, -1)
                    
                    # Predict
                    pred_label = ml_model["model"].predict(feat)[0]
                    pred_proba = ml_model["model"].predict_proba(feat)[0]
                    
                    # Get class names
                    class_names = ml_model["label_encoder"].classes_
                    
                    st.success(f"**Predicted Class:** {pred_label}")
                    
                    # Show probabilities
                    prob_df = pd.DataFrame({
                        "Class": class_names,
                        "Confidence": pred_proba
                    }).sort_values("Confidence", ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(prob_df["Class"], prob_df["Confidence"], color="#2ecc71")
                    ax.set_xlabel("Confidence Score")
                    ax.set_title("Classification Probabilities (ML Model)")
                    ax.set_xlim(0, 1)
                    st.pyplot(fig)
                else:
                    st.error("ML model not found. Please train the model first.")
            
            else:  # Deep Learning
                dl_model = load_dl_model()
                if dl_model is not None:
                    # Preprocess for DL
                    img_resized = cv2.resize(image_bgr, (224, 224))
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_array = np.expand_dims(img_rgb, axis=0)
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                    
                    # Predict
                    predictions = dl_model.predict(img_array, verbose=0)
                    pred_idx = np.argmax(predictions[0])
                    pred_confidence = predictions[0][pred_idx]
                    
                    # Get class names
                    class_names = load_dataset_info()
                    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
                    
                    st.success(f"**Predicted Class:** {pred_label}")
                    st.metric("Confidence Score", f"{pred_confidence:.2%}")
                    
                    # Show probabilities
                    prob_df = pd.DataFrame({
                        "Class": class_names,
                        "Confidence": predictions[0]
                    }).sort_values("Confidence", ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(prob_df["Class"], prob_df["Confidence"], color="#3498db")
                    ax.set_xlabel("Confidence Score")
                    ax.set_title("Top 10 Classification Probabilities (DL Model)")
                    ax.set_xlim(0, 1)
                    st.pyplot(fig)
                    
                    # Grad-CAM Visualization
                    st.subheader("Explainable AI: Grad-CAM Heatmap")
                    st.markdown("""
                    The heatmap below shows which regions of the leaf the model focuses on when making its prediction.
                    Red areas indicate high importance, blue areas indicate low importance.
                    """)
                    
                    try:
                        heatmap = get_gradcam_heatmap(
                            dl_model, 
                            img_array, 
                            "mobilenet_v2_base",
                            pred_index=pred_idx
                        )
                        superimposed = display_gradcam(img_rgb, heatmap, alpha=0.5)
                        st.image(superimposed, caption="Grad-CAM Heatmap Overlay", use_column_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate Grad-CAM: {e}")
                else:
                    st.error("DL model not found. Please train the model first.")

# ==================== MODEL COMPARISON ====================
elif page == "📊 Model Comparison":
    st.header("ML vs DL Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classical ML (Random Forest + SVM)")
        st.markdown("""
        **Advantages:**
        - Fast training and inference
        - Interpretable features
        - Requires less computational power
        - Works well with limited data
        
        **Disadvantages:**
        - Manual feature engineering required
        - May miss complex patterns
        - Requires careful preprocessing
        """)
    
    with col2:
        st.subheader("Deep Learning (MobileNetV2)")
        st.markdown("""
        **Advantages:**
        - Automatic feature learning
        - Handles complex patterns
        - Transfer learning leverages pre-trained knowledge
        - Better for large datasets
        
        **Disadvantages:**
        - Requires more computational power
        - Needs more training data
        - Longer training time
        - Black box nature (solved with Grad-CAM)
        """)
    
    # Load metrics if available
    try:
        ml_metrics_path = Path("outputs/notebook_run/ml_metrics.csv")
        if ml_metrics_path.exists():
            ml_metrics = pd.read_csv(ml_metrics_path)
            st.subheader("ML Model Metrics")
            st.dataframe(ml_metrics, use_container_width=True)
    except:
        pass
    
    st.markdown("---")
    st.info("💡 **Recommendation:** Use Classical ML for quick prototyping and interpretability. Use DL for production with Grad-CAM for explainability.")

# ==================== DATASET EXPLORER ====================
elif page == "📈 Dataset Explorer":
    st.header("Dataset Explorer")
    
    classes = load_dataset_info()
    
    if classes:
        st.subheader(f"Available Classes ({len(classes)})")
        
        # Display the classes loaded from the artifact model cache (since TFDS handles raw files)
        st.info("The original project loaded local files for Explorer, but it is now linked fully to abstract TensorFlow Datasets.")
        
        st.markdown("### Discovered Classes:")
        for cls in classes:
            st.markdown(f"- {cls}")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 2rem; background: rgba(255,255,255,0.02); border-radius: 1rem; margin-top: 3rem;'>
    <h4 style='color: #34d399; margin-bottom: 0.5rem;'>🌿 Plant Disease Detection System</h4>
    <p style='margin: 0;'>Built with Streamlit • Machine Learning • Deep Learning • Explanations (TFDS)</p>
    <p style='margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.7;'>Models: Random Forest, SVM, MobileNetV2</p>
</div>
""", unsafe_allow_html=True)
