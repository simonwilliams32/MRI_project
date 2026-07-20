from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import joblib
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
import io
import numpy as np
import uvicorn


app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "simonwilliams_16_13_39.joblib"
model = joblib.load(MODEL_PATH)

app.state.model_seq = load_model("saved_models/my_model.keras")
sequential_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']


@app.get("/")
def index():
    return {"ok": True, "message": "API is working"}

@app.get("/predict")
def predict(age: float,  sex: str, bmi: float, tumour_volume: float, growth_rate: float, laterality: str, comorbidity_index: float):
    patient = pd.DataFrame([{
        "BMI": bmi,
        "Scan 1: Volume": tumour_volume,
        "linear_regression_growth_rate_cm3": growth_rate,
        "charlson_comorbidity_index": comorbidity_index,
        "age_at_diagnosis": age,
        "Sex": sex,
        "tumour_laterality": laterality
    }])

    prediction = model.predict(patient)[0]

    return {"suggested_management_strategy": str(prediction)}


#After this, i ran uvicorn api.main:app --reload in my terminal at root MRI_project
"""
output was this:
➜  MRI_project git:(simonebranch3) ✗ uvicorn api.main:app --reload
INFO:     Will watch for changes in these directories: ['/Users/simonwilliams/code/simonwilliams32/MRI_project']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [29431] using StatReload

This is my ReDock format http://127.0.0.1:8000/redoc
Used this for testing http://127.0.0.1:8000/docs
"""


"""I then set up a .dockerignore file so it doesnt upload the data; made a requirements.txt file;
and created a dockerfile with the basics. Then built the docker docker build -t mri-api ; then ran it docker run -p 8080:8000 mri-api
then opened it http://127.0.0.1:8080/docs
"""

@app.post("/predict_sequential")
async def predict_image(file: UploadFile = File(...)):
    # 1. Read and decode the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")

    # 2. Preprocess for CNN (e.g., resize to 224x224 and normalize)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # 3. Run prediction
    prediction = app.state.model_seq.predict(image_array)
    result = np.argmax(prediction)
    label = sequential_classes[result]
    probas = {sequential_classes[i]: float(prediction[0][i]) for i in range(len(sequential_classes))}
    return {"Sequential_Prediction": float(result),
            "Sequential_Label": label,
            "Sequential_Proba": probas}


@app.get("/get_image")
async def get_image():
    image_path = Path(".jpg")
    if not image_path.is_file():
        return {"error": "Image not found"}
    return FileResponse(image_path)
