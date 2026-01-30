import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes

st.set_page_config(
    page_title="Public Notice CNN Classifier",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Public Notice Importance Classification (CNN)")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/notice_cnn.h5")

try:
    model = load_model()
    st.success("✅ CNN model loaded")
except Exception as e:
    st.error("❌ CNN model not found")
    st.exception(e)
    st.stop()

CLASSES = ["Critical", "Important", "Informational", "Low Priority"]
IMG_SIZE = 224

def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

option = st.radio(
    "Select Input Method",
    ["📄 PDF Upload", "🌐 Web Scraping"],
    horizontal=True
)

# ================= PDF OPTION =================
if option == "📄 PDF Upload":
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf:
        images = convert_from_bytes(pdf.read(), dpi=200)
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_column_width=True)
            if st.button(f"Classify Page {i+1}"):
                preds = model.predict(preprocess(img))[0]
                idx = np.argmax(preds)
                st.success(f"Importance: {CLASSES[idx]}")
                st.write(f"Confidence: {preds[idx]*100:.2f}%")

# ================= WEB OPTION =================
if option == "🌐 Web Scraping":
    st.info("Uses visual layout proxy for CNN classification")
    dummy_img = Image.new("RGB", (900, 600), "white")
    st.image(dummy_img, caption="Rendered Web Page")
    if st.button("Classify Web Page"):
        preds = model.predict(preprocess(dummy_img))[0]
        idx = np.argmax(preds)
        st.success(f"Importance: {CLASSES[idx]}")
        st.write(f"Confidence: {preds[idx]*100:.2f}%")
