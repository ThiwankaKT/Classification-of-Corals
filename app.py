# ==============================================================
# app.py — Streamlit Web App for Coral Health Prediction
# ==============================================================

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

# ==============================================================
# 1. PAGE CONFIGURATION
# ==============================================================

st.set_page_config(
    page_title="Coral Health Classifier 🌊",
    page_icon="🌺",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #e8f4f8;
        color: #00334d;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton button {
        background-color: #006994;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #00aaff;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================
# 2. MODEL LOADING
# ==============================================================

@st.cache_resource
def load_effnet_model():
    try:
        model = load_model("model_effnet.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_effnet_model()

if model is None:
    st.stop()

# ==============================================================
# 3. IMAGE PREPROCESSING FUNCTION
# ==============================================================

def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ==============================================================
# 4. APP UI
# ==============================================================

st.title("🐠 Coral Health Classification 🌊")
st.markdown(
    """
    Upload a coral image, and this app will predict whether it’s **Healthy** or **Bleached**  
    using a fine‑tuned **EfficientNetB0** model.
    """
)

uploaded_file = st.file_uploader("📸 Upload a coral image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    img_data = preprocess_image(image)

    if st.button("🔍 Predict Health Status"):
        with st.spinner("Analyzing coral image..."):
            preds = model.predict(img_data)
            confidence = float(preds[0][0])
            is_healthy = confidence < 0.5

            predicted_label = "Healthy Coral 🪸" if is_healthy else "Bleached Coral ⚪"
            probability = 1 - confidence if is_healthy else confidence

            st.subheader(f"Prediction: **{predicted_label}**")
            st.progress(int(probability * 100))
            st.write(f"**Confidence:** {probability * 100:.2f}%")

            # Optional note
            st.markdown(
                "*(Confidence closer to 100% means the model is more certain about its prediction.)*"
            )

# ==============================================================
# 5. FOOTER
# ==============================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #1a5276'>
        🌺 Developed with TensorFlow & Streamlit · Coral Health Detection Project
    </div>
    """,
    unsafe_allow_html=True
)
