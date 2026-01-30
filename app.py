import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ----------------
IMG_SIZE = 224
CLASSES = ["Critical", "Important", "Informational", "Low Priority"]

# ---------------- LOAD CNN MODEL ----------------
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("model/notice_cnn.h5")

model = load_cnn_model()

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI ----------------
st.set_page_config(page_title="CNN Public Notice Classifier")
st.title("📰 CNN-Based Public Notice Importance Classification")

st.markdown("""
**Core Model:** Convolutional Neural Network (Image Classification)  
**Input:** Public Notice Image (Gujarati / English)  
**Output:** Importance Level
""")

file = st.file_uploader("Upload Notice Image", ["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Notice", use_column_width=True)

    if st.button("Classify using CNN"):
        with st.spinner("CNN analyzing visual layout..."):
            x = preprocess(img)
            preds = model.predict(x)[0]
            idx = np.argmax(preds)

            st.success(f"📌 Importance: **{CLASSES[idx]}**")
            st.write(f"Confidence: **{preds[idx]*100:.2f}%**")

            st.subheader("Class Probabilities")
            for i, c in enumerate(CLASSES):
                st.write(f"{c}: {preds[i]*100:.2f}%")
else:
    st.info("Upload a public notice image to start.")
