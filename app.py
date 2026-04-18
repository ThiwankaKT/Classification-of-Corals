# ==============================================================
# Coral Health Classifier — Streamlit App (Refined UI)
# ==============================================================

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================================================
# 1. PAGE CONFIGURATION
# ==============================================================

st.set_page_config(
    page_title="Coral Health Classifier 🌊",
    page_icon="🐠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==============================================================
# 2. SIDEBAR
# ==============================================================

st.sidebar.title("📘 How to Use")
st.sidebar.markdown(
    """
    1. Upload a coral image (`.jpg`, `.png`, `.jpeg`)
    2. Click **Predict Health Status**
    3. The app shows if the coral is *Healthy* 🪸 or *Bleached* ⚪  
       along with the model’s confidence level.

    ---
    ### 🧠 About the Model
    - Backbone: **EfficientNetB0**
    - Dataset: Healthy vs Bleached Corals  
    - Framework: TensorFlow/Keras
    """
)

# ==============================================================
# 3. GLOBAL PAGE STYLE
# ==============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #e8f4f8;
        }
        .stButton button {
            background-color: #0077a8;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #009fd4;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================
# 4. LOAD MODEL
# ==============================================================

@st.cache_resource
def load_effnet_model():
    try:
        model = load_model("model_effnet.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_effnet_model()
if model is None:
    st.stop()

# ==============================================================
# 5. IMAGE PREPROCESSING
# ==============================================================

def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ==============================================================
# 6. MAIN CONTENT
# ==============================================================

st.title("🐠 Coral Health Classification 🌊")
st.markdown(
    """
    Upload a coral image, and this app will predict whether it’s  
    **Healthy** 🪸 or **Bleached** ⚪ using a fine‑tuned **EfficientNetB0** model.
    """
)

uploaded_file = st.file_uploader("📸 Upload a coral image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Coral Image", width="stretch")

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
# 7. FOOTER
# ==============================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #006680; font-size: 14px;'>
        🌺 Developed using TensorFlow & Streamlit · Coral Health Detection Project
    </div>
    """,
    unsafe_allow_html=True
)
