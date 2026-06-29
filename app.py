import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Car Crash Detection", page_icon="🚗", layout="centered")

# ----------------------------
# CUSTOM CSS (LIGHT BLUE THEME)
# ----------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #eaf4ff;
}

h1 {
    color: #1e3a8a;
    text-align: center;
}

p {
    color: #475569;
    text-align: center;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# TITLE
# ----------------------------
st.markdown("<h1>🚗 Car Crash Severity Detection</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload an image to analyze accident severity</p>", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_my_model():
    return load_model("car_crash_model.h5")

with st.spinner("🔄 Loading AI model..."):
    model = load_my_model()

classes = ["Minor", "Moderate", "Major"]

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 👇 FIXED IMAGE SIZE
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(img, caption="Uploaded Image", width=400)
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------
    # PREPROCESS
    # ----------------------------
    img_resized = cv2.resize(img, (224,224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # ----------------------------
    # PREDICTION
    # ----------------------------
    prediction = model.predict(img_resized)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    # ----------------------------
    # CONCLUSION
    # ----------------------------
    if predicted_class == "Minor":
        color = "#16a34a"
        conclusion = "Low severity crash. Minor damage, safe condition."
    elif predicted_class == "Moderate":
        color = "#f59e0b"
        conclusion = "Moderate crash detected. Possible vehicle damage."
    else:
        color = "#dc2626"
        conclusion = "High severity crash detected. Immediate action required."

    # ----------------------------
    # OUTPUT
    # ----------------------------
    st.markdown(f"""
    <div class='card'>
        <h3 style='color:{color};'>Prediction: {predicted_class}</h3>
        <h4>Confidence: {confidence*100:.2f}%</h4>
        <p>{conclusion}</p>
    </div>
    """, unsafe_allow_html=True)