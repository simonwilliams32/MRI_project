import numpy as np
from PIL import Image
from tensorflow.keras.applications import efficientnet
import cv2
import keras

IMG_SIZE = (224, 224)

def preprocess_image(img):
    """
    this function is used for the processing of the image provided by the user
    """

    #img = Image.open(image_path).convert("L")

    # resize
    #img = img.convert("L")
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    # convert to np.array
    arr = np.array(img).astype(np.float32)

    # scale using the efficentnet
    arr = efficientnet.preprocess_input(arr)

    # Add batch size (1)
    arr = np.expand_dims(arr, axis=0)
    arr = arr[0:1,:,:,0:1]
    print(arr.shape)

    return arr


def get_model():
    binary_model = keras.saving.load_model("saved_models/binary_model2.keras", safe_mode=False)
    return binary_model
