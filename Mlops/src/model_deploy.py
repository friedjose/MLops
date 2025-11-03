# model_deploy.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================
# Load Best Model
# ==========================
model = joblib.load("best_model.pkl")

# ==========================
# Initialize FastAPI App
# ==========================
app = FastAPI(
    title="API de Predicción Membresía Premium - Restaurantes",
    description="Modelo para predecir si un cliente adquiere membresía premium",
    version="1.0"
)

# ==========================
# Esquema entrada 1 registro
# ==========================
class InputData(BaseModel):
    features: List[float]   # valores ya transformados por pipeline


@app.post("/predict_one")
def predict_one(data: InputData):
    df = pd.DataFrame([data.features])
    pred = model.predict(df)[0]

    # Si el modelo soporta predict_proba
    try:
        prob = model.predict_proba(df)[0].tolist()
    except:
        prob = None

    return {
        "prediction": int(pred),
        "membership": "Premium" if pred == 1 else "No Premium",
        "probabilities": prob
    }


# ==========================
# Esquema entrada batch
# ==========================
class BatchData(BaseModel):
    batch: List[List[float]]


@app.post("/predict_batch")
def predict_batch(data: BatchData):
    df = pd.DataFrame(data.batch)
    preds = model.predict(df).tolist()

    # Probabilidades si disponible
    try:
        probs = model.predict_proba(df).tolist()
    except:
        probs = None

    return {
        "predictions": preds,
        "membership_labels": ["Premium" if p == 1 else "No Premium" for p in preds],
        "probabilities": probs
    }
