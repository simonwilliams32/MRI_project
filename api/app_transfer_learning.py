import io
import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import efficientnet

# config section
IMG_SIZE = (224, 224)
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

MAIN_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = os.path.join(MAIN_DIR, "saved_models")
BINARY_MODEL_PATH = os.path.join(MODELS_DIR, "brain_tumor_model_binary_transferlearning.keras")
MULTI_MODEL_PATH = os.path.join(MODELS_DIR, "brain_tumor_model_4class_transferlearning.keras")

BINARY_CLASS_NAMES = ["healthy", "tumour detected"]
MULTI_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# load models
app = FastAPI()
binary_model = tf.keras.models.load_model(BINARY_MODEL_PATH)
multi_model = tf.keras.models.load_model(MULTI_MODEL_PATH)

# health check
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "binary_model_loaded": binary_model is not None,
        "multi_model_loaded": multi_model is not None,
    }


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Mirrors the exact preprocessing used at training time."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr = efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.post("/predict/binary")
async def predict_binary(file: UploadFile = File(...)):
    ### Receiving and decoding the image
    image_bytes = await file.read()

    x = preprocess_image(image_bytes)

    prob_cancer = float(binary_model.predict(x, verbose=0)[0][0])
    predicted_class = "Tumour detected" if prob_cancer > 0.5 else "healthy"

    return {
        "predicted_class": predicted_class,
        "probability_tumour": round(prob_cancer, 4),
        "probability_healthy": round(1 - prob_cancer, 4),
    }

@app.post("/predict/four-class")
async def predict_four_class(file: UploadFile = File(...)):
    ### Receiving and decoding the image
    image_bytes = await file.read()

    x = preprocess_image(image_bytes)

    preds = multi_model.predict(x, verbose=0)[0]
    predicted_idx = int(np.argmax(preds))

    return {
        "predicted_class": MULTI_CLASS_NAMES[predicted_idx],
        "confidence": round(float(preds[predicted_idx]), 4),
        "all_probabilities": {
            name: round(float(p), 4) for name, p in zip(MULTI_CLASS_NAMES, preds)
        },
    }


'''➜  MRI_project git:(main) ✗ uvicorn api.app_transfer_learning:app --reload --port 8001                                                                          [🐍 MRI_project]
INFO:     Will watch for changes in these directories: ['/Users/simonwilliams/code/simonwilliams32/MRI_project']
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
INFO:     Started reloader process [4437] using StatReload'''
