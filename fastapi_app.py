from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Attack Risk Prediction API!"}

class HeartData(BaseModel):
    Age: float
    Sex: int
    ChestPainType: int
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    RestingECG: int
    MaxHR: float
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int

model_path = os.path.join("models", "best_model.joblib")
model = joblib.load(model_path)

@app.post("/predict")
def predict(data: HeartData):
    input_array = np.array([[
        data.Age,
        data.Sex,
        data.ChestPainType,
        data.RestingBP,
        data.Cholesterol,
        data.FastingBS,
        data.RestingECG,
        data.MaxHR,
        data.ExerciseAngina,
        data.Oldpeak,
        data.ST_Slope
    ]])

    prediction_proba = model.predict_proba(input_array)[0][1]
    prediction = model.predict(input_array)[0]

    return {
        "prediction": int(prediction),
        "risk_probability": f"{prediction_proba:.2%}"
    }


