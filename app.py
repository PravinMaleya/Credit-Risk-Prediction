import json

import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("models/xgb_model.joblib")
preprocessor = joblib.load("models/credit_risk_preprocessor.joblib")

with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])

    processed_data = preprocessor.transform(input_df)

    probability = model.predict_proba(processed_data)[0, 1]

    prediction = int(probability >= threshold)

    return {
        "default_probability": float(probability),
        "threshold": threshold,
        "prediction": prediction,
        "result": "default" if prediction == 1 else "non-default"
    }