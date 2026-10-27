import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

saved = joblib.load(Path(__file__).resolve().parent/"risk_model.pkl")
app = FastAPI(title="Diabetes Risk API (learning project)")

from pydantic import BaseModel, Field

class Patient(BaseModel):
    bmi: float = Field(ge=15, le=40.8)
    hba1c_level: float = Field(ge=4.8, le=13.4)
    age: int = Field(ge=18, le=80)
    fasting_blood_sugar: float = Field(ge=73, le=401)
    waist_circumference_cm: float = Field(ge=73.2, le=148.2)

@app.post("/predict")
def predict(patient: Patient):
    row = pd.DataFrame([patient.model_dump()])[saved["features"]]
    p = float(saved["model"].predict_proba(row)[0, 1])
    return {
        "probability": round(p, 3),
        "flagged": p >= saved["threshold"],
        "threshold": saved["threshold"],
    }
