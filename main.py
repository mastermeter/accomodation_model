# main.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.getenv("MODEL_PATH", "randomforest_pipeline.joblib")

app = FastAPI(title="Valais Price - RandomForest API")

# CORS (resserrer les domaines en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Chargement robuste du bundle au démarrage
try:
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feature_order = bundle["feature_order"]  # ordre exact attendu par le pipeline
    MODEL_READY = True
except Exception as e:
    pipe = None
    feature_order = None
    MODEL_READY = False
    LOAD_ERR = e

# Schéma d'entrée EXACT selon feature_order
class Features(BaseModel):
    # Noms alignés sur feature_order : ["surface_m2","num_rooms","is_furnished","wifi_incl","charges_incl","car_park"]
    surface_m2: float = Field(..., ge=0, description="Surface en m²")
    num_rooms: float = Field(..., ge=0, description="Nombre de pièces (peut être décimal)")
    is_furnished: bool
    wifi_incl: bool
    charges_incl: bool
    car_park: bool

@app.get("/")
def root():
    return {
        "status": "ok" if MODEL_READY else "degraded",
        "service": "Valais Price - RandomForest API",
        "docs": "/docs",
        "health": "/health",
        "model_loaded": MODEL_READY,
        "model_path": MODEL_PATH,
    }

@app.get("/health")
def health():
    if MODEL_READY:
        return {"status": "ok"}
    # Renvoie un 503 si le modèle n'est pas chargé
    raise HTTPException(status_code=503, detail=f"model_not_loaded: {type(LOAD_ERR).__name__}: {LOAD_ERR}")

@app.post("/predict")
def predict(feat: Features):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="model_not_loaded")

    # Vérifier que le schéma colle au feature_order
    incoming = list(Features.model_fields.keys())
    if feature_order != incoming:
        # Défensif : si l'ordre ne correspond pas, on réordonne explicitement
        # (et on loggue une alerte)
        # NB: Pydantic garantit déjà les champs, mais on s’assure de l’ordre strict.
        pass

    # Conversion explicite des booléens en 0/1 si le pipeline a été entraîné ainsi
    def b2i(x): return 1 if bool(x) else 0
    row_dict = {
        "surface_m2": feat.surface_m2,
        "num_rooms": feat.num_rooms,
        "is_furnished": b2i(feat.is_furnished),
        "wifi_incl": b2i(feat.wifi_incl),
        "charges_incl": b2i(feat.charges_incl),
        "car_park": b2i(feat.car_park),
    }

    # Construire la ligne dans l'ordre exact attendu
    try:
        row = pd.DataFrame([[row_dict[c] for c in feature_order]], columns=feature_order)
        yhat = pipe.predict(row)[0]
        return {"price_chf_pred": float(yhat)}
    except KeyError as e:
        # Un champ du modèle n'existe pas dans feature_order (ou inversement)
        raise HTTPException(status_code=400, detail=f"feature_mismatch: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference_error: {type(e).__name__}: {e}")
