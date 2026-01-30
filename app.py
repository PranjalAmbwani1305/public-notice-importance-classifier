import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pdf2image import convert_from_bytes
import os
from web_scraping import scrape_news

MODEL_PATH = "model/notice_cnn.h5"
IMG_SIZE = 224
CLASSES = ["Critical", "Important", "Informational", "Low Priority"]

st.set_page_config(page_title="Public Notice CNN")
st.title("📰 Public Notice Importance Classification")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Train the CNN first using cnn_train.py")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
st.success("✅ CNN model loaded")

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

# ---------- PDF ----------
if option == "📄 PDF Upload":
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf:
        images = convert_from_bytes(pdf.read(), dpi=200)
        for img in images:
            st.image(img, use_column_width=True)
            preds = model.predict(preprocess(img))[0]
            st.success(f"Importance: {CLASSES[np.argmax(preds)]}")

# ---------- WEB ----------
if option == "🌐 Web Scraping":
    if st.button("Fetch & Classify BBC + Indian Express"):
        text = scrape_news()

        img = Image.new("RGB", (900, 600), "white")
        draw = ImageDraw.Draw(img)
        y = 10
        for line in text.split("."):
            draw.text((10, y), line[:120], fill="black")
            y += 20
            if y > 580:
                break

        st.image(img, caption="Rendered News Image", use_column_width=True)
        preds = model.predict(preprocess(img))[0]
        st.success(f"Importance: {CLASSES[np.argmax(preds)]}")
