# main.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Charger pipeline + meta
bundle = joblib.load("randomforest_pipeline.joblib")
pipe = bundle["pipeline"]
feature_order = bundle["feature_order"]  # ordre exact des colonnes

app = FastAPI(title="Valais Price - RandomForest API")

# CORS (restreins en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Schéma d'entrée EXACT selon feature_order
class Features(BaseModel):
    surface_m2: float
    num_rooms: float
    is_furnished: bool
    wifi_incl: bool
    charges_incl: bool
    car_park: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(feat: Features):
    row = pd.DataFrame([[getattr(feat, c) for c in feature_order]], columns=feature_order)
    yhat = pipe.predict(row)[0]
    return {"price_chf_pred": float(yhat)}
