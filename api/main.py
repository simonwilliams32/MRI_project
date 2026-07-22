from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI



app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "simonwilliams_16_13_39.joblib"
model = joblib.load(MODEL_PATH)


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

    probabilities = model.predict_proba(patient)[0]
    classes = model.classes_

    class_probabilities = {str(class_name): float(probability)
        for class_name, probability in zip(classes, probabilities)}

    prediction_confidence = class_probabilities[str(prediction)]

    return {"suggested_management_strategy": str(prediction),
        "confidence": prediction_confidence,
        "class_probabilities": class_probabilities,}


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

API then pushed to google cloud, available at https://mri-api-677473747782.europe-west2.run.app
"""
