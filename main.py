import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "model/notice_cnn.h5"
IMG_SIZE = 224
CLASSES = ["Critical", "Important", "Informational", "Low Priority"]

st.set_page_config(page_title="CNN Public Notice Classifier")
st.title("📰 Public Notice Importance Classification (CNN)")

# ===============================
# ENSURE MODEL EXISTS
# ===============================
def create_dummy_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)

# ===============================
# LOAD CNN MODEL SAFELY
# ===============================
@st.cache_resource
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        st.warning("CNN model not found. Creating a dummy CNN model...")
        create_dummy_model()

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_safe()
st.success("✅ CNN model loaded successfully")

# ===============================
# IMAGE PREPROCESS
# ===============================
def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ===============================
# UI
# ===============================
uploaded_file = st.file_uploader(
    "Upload Public Notice Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True)

    if st.button("Classify Notice"):
        preds = model.predict(preprocess(img))[0]
        idx = np.argmax(preds)

        st.success(f"📌 Importance: {CLASSES[idx]}")
        st.write(f"Confidence: {preds[idx]*100:.2f}%")
else:
    st.info("Upload an image to start CNN classification.")
