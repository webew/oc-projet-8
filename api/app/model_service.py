import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import json

# Dossier api/ (stable, même si on lance pytest depuis ailleurs)
API_DIR = Path(__file__).resolve().parents[1]  # .../api
MODEL_PATH = API_DIR / "models" / "model.pkl"

# Récupération du seuil métier
EXPORT_DATA_FILE = API_DIR / "models" / "export_data.json"
# Lecture du seuil métier depuis le fichier
with open(EXPORT_DATA_FILE, "r") as f:
    threshold_data = json.load(f)
    THRESHOLD = float(threshold_data["threshold"])

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_from_features(features: Dict[str, Any]) -> Tuple[bool, float]:
    model = get_model()

    cols = list(model.feature_names_in_)
    X = pd.DataFrame([{c: features.get(c) for c in cols}], columns=cols)

    if hasattr(model, "predict_proba"):
        proba_default = float(model.predict_proba(X)[0][1])
    else:
        pred = float(model.predict(X)[0])

    approved = proba_default < float(THRESHOLD)
    return approved, proba_default, THRESHOLD
