import io
import os
import numpy as np
import tensorflow as tf
import keras
import matplotlib as mpl
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import Response
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications import efficientnet

# config section
IMG_SIZE = (224, 224)
#ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"} # this can be implemented later

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

# preprocessing function
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Use the exact preprocessing used at training time."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr = efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


# binary prediction model
@app.post("/predict/binary")
async def predict_binary(file: UploadFile = File(...)):

    image_bytes = await file.read()

    x = preprocess_image(image_bytes)

    prob_cancer = float(binary_model.predict(x, verbose=0)[0][0])
    predicted_class = "Tumour detected" if prob_cancer > 0.5 else "healthy"

    return {
        "predicted_class": predicted_class,
        "probability_tumour": round(prob_cancer, 4),
        "probability_healthy": round(1 - prob_cancer, 4),
    }


# multi-class prediction model
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

##### GRAD-CAM implementation

# helper functions

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    image_tensor = tf.Variable(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = image_tensor
        last_conv_layer_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                last_conv_layer_output = x
                tape.watch(last_conv_layer_output)
        preds = x
        class_channel = preds[:, 0]  # binary sigmoid: single output unit

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    img_array = keras.utils.img_to_array(original_img)

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed = jet_heatmap * alpha + img_array * (1 - alpha)
    return keras.utils.array_to_img(superimposed)

# define endpoint
@app.post("/predict/binary/gradcam")
async def predict_binary_gradcam(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = preprocess_image(image_bytes)

    prob_cancer = float(binary_model.predict(x, verbose=0)[0][0])
    predicted_class = "Tumour detected" if prob_cancer > 0.5 else "healthy"

    # change the last_conv_layer_name if another model is used
    heatmap = make_gradcam_heatmap(x, binary_model, last_conv_layer_name="efficientnetb0")

    original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    overlay_img = overlay_gradcam(original_img, heatmap, alpha=0.4)

    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "X-Predicted-Class": predicted_class,
            "X-Probability-Tumour": f"{prob_cancer:.4f}",
        },
    )
'''➜  MRI_project git:(main) ✗ uvicorn api.app_transfer_learning:app --reload --port 8001                                                                          [🐍 MRI_project]
INFO:     Will watch for changes in these directories: ['/Users/simonwilliams/code/simonwilliams32/MRI_project']
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
INFO:     Started reloader process [4437] using StatReload'''
