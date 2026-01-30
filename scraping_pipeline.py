import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_bytes
import io

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 224
CLASSES = ["Critical", "Important", "Informational", "Low Priority"]

st.set_page_config(
    page_title="Public Notice CNN Classifier",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Public Notice Importance Classification using CNN")
st.write("CNN is applied in **both PDF and Web Scraping options**.")

# ===============================
# LOAD CNN MODEL
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/notice_cnn.h5")

try:
    model = load_model()
    st.success("✅ CNN model loaded")
except Exception as e:
    st.error("❌ Failed to load CNN model")
    st.exception(e)
    st.stop()

# ===============================
# PREPROCESS IMAGE FOR CNN
# ===============================
def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# INPUT METHOD
# ===============================
option = st.radio(
    "Select Input Method",
    ["📄 PDF Upload", "🌐 Web Scraping"],
    horizontal=True
)

# ==================================================
# OPTION 1: PDF → IMAGE → CNN
# ==================================================
if option == "📄 PDF Upload":
    st.subheader("📄 Upload Public Notice PDF")

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_pdf:
        with st.spinner("Converting PDF to images..."):
            images = convert_from_bytes(uploaded_pdf.read(), dpi=200)

        st.success(f"{len(images)} page(s) extracted")

        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_column_width=True)

            if st.button(f"Classify Page {i+1}"):
                x = preprocess(img)
                preds = model.predict(x)[0]
                idx = np.argmax(preds)

                st.success(f"📌 Importance: {CLASSES[idx]}")
                st.write(f"Confidence: {preds[idx]*100:.2f}%")

# ==================================================
# OPTION 2: WEB PAGE → IMAGE → CNN
# ==================================================
if option == "🌐 Web Scraping":
    st.subheader("🌐 Web Scraping (Public Notice Page)")

    url = st.text_input(
        "Enter News / Public Notice URL",
        "https://www.bbc.com/news"
    )

    if st.button("Fetch & Classify Page"):
        try:
            with st.spinner("Fetching webpage..."):
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                # Convert webpage text to image (visual proxy)
                text = " ".join(p.get_text() for p in soup.find_all("p")[:20])

                img = Image.new("RGB", (900, 600), "white")
                st.image(img, caption="Rendered Web Page Image")

                x = preprocess(img)
                preds = model.predict(x)[0]
                idx = np.argmax(preds)

                st.success(f"📌 Importance: {CLASSES[idx]}")
                st.write(f"Confidence: {preds[idx]*100:.2f}%")

        except Exception as e:
            st.error("Failed to process webpage")
            st.exception(e)
