import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json

# -------------------------------
# Load different models
# -------------------------------
models = {
    "Simple CNN": "SimpleCNN.h5",
    "U-Net": "UNet.h5",
    "GAN": "GAN.h5"
    
}

# -------------------------------
# Load precomputed metrics
# -------------------------------
with open("model_metrics.json", "r") as f:
    model_metrics = json.load(f)

# Find best model (highest SSIM, then PSNR as tie-breaker)
best_model = max(model_metrics.items(), key=lambda kv: (kv[1]["SSIM"], kv[1]["PSNR"]))[0]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Grayscale to RGB Colorization", layout="wide")

st.title("🎨 Grayscale → RGB Colorization")
st.markdown(
    """
    Upload a **grayscale image** and choose a model to predict the RGB version.  
    Optionally, upload a ground-truth image to compare results.
    """
)

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Choose a model:", list(models.keys()))
    st.caption("Switch models to compare different approaches.")

    st.markdown("### 📊 Model Evaluation")
    for name, metrics in model_metrics.items():
        label = f"⭐ {name}" if name == best_model else name
        st.markdown(
            f"- **{label}**  \n"
            f"  Average SSIM: {metrics['SSIM']:.4f}  \n"
            f"  Average PSNR: {metrics['PSNR']:.2f} dB"
        )

    st.success(f"🏆 Best Model: **{best_model}** (highest SSIM/PSNR)")

# -------------------------------
# Load selected model
# -------------------------------
@st.cache_resource
def load_selected_model(path):
    return load_model(path, compile=False)

model = load_selected_model(models[model_choice])

# -------------------------------
# File uploaders
# -------------------------------
col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    gray_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])
with col_upload2:
    rgb_file  = st.file_uploader("Upload ground truth RGB (optional)", type=["jpg", "jpeg", "png"])

if gray_file is not None:
    # -------------------------------
    # Preprocess Grayscale Input
    # -------------------------------
    gray_img = Image.open(gray_file).convert("L")   # force grayscale
    img_resized = gray_img.resize((32, 32))  # adjust to training input size
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)   # (32,32,1)
    img_array = np.expand_dims(img_array, axis=0)    # (1,32,32,1)

    # -------------------------------
    # Predict
    # -------------------------------
    with st.spinner(f"Running {model_choice}..."):
        pred = model.predict(img_array)[0]
    pred = np.clip(pred, 0, 1)

    # -------------------------------
    # Display results
    # -------------------------------
    st.subheader(f"🖼️ Results - {model_choice}")

    if rgb_file is not None:
        gt_img = Image.open(rgb_file).convert("RGB").resize((32, 32))
        gt_array = np.array(gt_img).astype("float32") / 255.0

        # Compute metrics
        pred_tf = tf.convert_to_tensor(np.expand_dims(pred, axis=0), dtype=tf.float32)
        gt_tf   = tf.convert_to_tensor(np.expand_dims(gt_array, axis=0), dtype=tf.float32)

        ssim_val = tf.image.ssim(gt_tf, pred_tf, max_val=1.0)
        psnr_val = tf.image.psnr(gt_tf, pred_tf, max_val=1.0)

        # Metrics display
        st.markdown("### 📊 Quality Metrics (for this image)")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.metric(label="SSIM", value=f"{ssim_val.numpy()[0]:.4f}")
        with mcol2:
            st.metric(label="PSNR", value=f"{psnr_val.numpy()[0]:.2f} dB")

        # Side-by-side comparison
        st.markdown("### 🔍 Side-by-Side Comparison")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(gray_img, caption="Grayscale Input", use_container_width=True)
        with col2:
            st.image(gt_array, caption="Ground Truth RGB", use_container_width=True)
        with col3:
            st.image(pred, caption=f"Predicted ({model_choice})", use_container_width=True)

    else:
        # Show grayscale vs prediction only
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_img, caption="Grayscale Input", use_container_width=True)
        with col2:
            st.image(pred, caption=f"Predicted ({model_choice})", use_container_width=True)
