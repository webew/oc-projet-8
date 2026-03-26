import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import json
import shap

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

# Charger l'importance globale
SHAP_GLOBAL_PATH = API_DIR / "models" / "shap_global.json"
with open(SHAP_GLOBAL_PATH) as f:
    SHAP_GLOBAL = json.load(f)

# Dans model_service.py, au chargement

X_scoring_path = API_DIR.parent / "data" / "processed" / "X_scoring.csv"
X_scoring_bg = pd.read_csv(X_scoring_path, index_col="SK_ID_CURR").sample(100, random_state=42)

def get_shap_local(X: pd.DataFrame) -> dict:
    model = get_model()
    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()

    # Background transformé
    X_bg = preprocess.transform(X_scoring_bg)
    X_bg_df = pd.DataFrame(
        X_bg.toarray() if hasattr(X_bg, "toarray") else X_bg,
        columns=feature_names
    )

    # Client transformé
    X_transformed = preprocess.transform(X)
    X_df = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
        columns=feature_names
    )

    # Explainer avec background
    explainer = shap.Explainer(clf, X_bg_df)
    shap_values = explainer(X_df, check_additivity=False)

    return dict(zip(feature_names, shap_values.values[0].tolist()))