import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Skin Cancer Image Classifier - Streamlit application
# ---------------------------------------------------------

st.set_page_config(
    page_title="Skin Cancer Image Classifier",
    page_icon="🔬",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "skin_cancer_detector.keras"
DATASET_PATHS = [
    APP_DIR / "dataset",
    APP_DIR / "data",
    APP_DIR / "unzipped_data" / "data",
]
IMG_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The original training code used flow_from_directory() with class_mode="binary".
# Keras assigns class indices in alphabetical order. For folders named
# "Benign" and "Malignant", index 0 = Benign and index 1 = Malignant.
CLASS_NAMES = ["Benign", "Malignant"]


@st.cache_resource
def load_model(model_path: str):
    """Load the trained Keras model once and cache it."""
    return tf.keras.models.load_model(model_path, compile=False)


def find_dataset_directory():
    """Return the first available dataset directory, if present."""
    for candidate in DATASET_PATHS:
        if candidate.is_dir():
            return candidate
    return None


def get_class_counts(dataset_dir: Path):
    """
    Count images in immediate class subfolders.
    Folders such as train, test and validation are not treated as classes.
    """
    counts = {}
    ignored = {"train", "test", "validation", "valid"}

    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir() or folder.name.lower() in ignored:
            continue

        image_count = sum(
            1 for file in folder.rglob("*")
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if image_count > 0:
            counts[folder.name] = image_count

    return counts


def preprocess_image(image: Image.Image):
    """Prepare an uploaded image exactly as the training pipeline expects."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


# ------------------------- Page heading -------------------------

st.title("🔬 Skin Cancer Image Classifier")
st.write(
    "Upload a skin image to receive a model prediction. "
    "This tool is for educational purposes only and is not a medical diagnosis."
)

# ------------------------- Load model -------------------------

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Put 'skin_cancer_detector.keras' "
        "in the same GitHub repository folder as this app."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("The model could not be loaded. Check the model file and TensorFlow version.")
    st.exception(error)
    st.stop()

# ------------------------- Sidebar / dataset overview -------------------------

st.sidebar.header("Dataset information")
dataset_dir = find_dataset_directory()

if dataset_dir is not None:
    class_counts = get_class_counts(dataset_dir)

    if class_counts:
        st.sidebar.success(f"Dataset found: {dataset_dir.name}")
        st.sidebar.write("Images found in class folders:")
        for class_name, count in class_counts.items():
            st.sidebar.write(f"- **{class_name}:** {count}")

        total_images = sum(class_counts.values())
        st.sidebar.metric("Total images", total_images)

        if total_images > 0:
            chart_col, _ = st.columns([1, 1])
            with chart_col:
                st.subheader("Dataset Class Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(
                    list(class_counts.values()),
                    labels=list(class_counts.keys()),
                    autopct="%1.1f%%",
                    startangle=140
                )
                ax.axis("equal")
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.sidebar.warning(
            "Dataset folder exists, but no images were found in its class subfolders. "
            "Image prediction can still work."
        )
else:
    st.sidebar.info(
        "No dataset folder found. Image prediction still works because the trained "
        "model is loaded separately."
    )

# ------------------------- Image prediction -------------------------

st.subheader("Upload an image")
uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        left_col, right_col = st.columns(2)

        with left_col:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with right_col:
            if st.button("Analyze Image", type="primary"):
                input_array = preprocess_image(image)
                prediction = model.predict(input_array, verbose=0)

                # The uploaded model ends with one sigmoid output.
                probability_class_1 = float(np.asarray(prediction).reshape(-1)[0])
                probability_class_1 = float(np.clip(probability_class_1, 0.0, 1.0))
                probability_class_0 = 1.0 - probability_class_1

                predicted_index = 1 if probability_class_1 >= 0.5 else 0
                predicted_label = CLASS_NAMES[predicted_index]
                confidence = (
                    probability_class_1 if predicted_index == 1
                    else probability_class_0
                )

                st.markdown("### Prediction")
                st.success(f"Predicted class: **{predicted_label}**")
                st.metric("Model confidence", f"{confidence * 100:.2f}%")

                st.write("Class probabilities")
                st.write(f"**{CLASS_NAMES[0]}:** {probability_class_0 * 100:.2f}%")
                st.progress(probability_class_0)
                st.write(f"**{CLASS_NAMES[1]}:** {probability_class_1 * 100:.2f}%")
                st.progress(probability_class_1)

                st.warning(
                    "This prediction is not a diagnosis. A qualified dermatologist "
                    "must assess any concerning or changing skin lesion."
                )

    except Exception as error:
        st.error("Could not process this image. Please upload a valid image file.")
        st.exception(error)

st.divider()
st.caption(
    "Educational AI project. Model predictions may be wrong and must not be used "
    "to make medical decisions."
)
