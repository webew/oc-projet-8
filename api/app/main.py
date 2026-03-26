import pandas as pd
from fastapi import FastAPI, HTTPException
from api.app.schemas import PredictRequest, PredictResponse
from api.app.model_service import predict_from_features, get_shap_local, SHAP_GLOBAL

app = FastAPI(title="OC Projet 7 - API", version="0.1.0")

# Chargé une seule fois au démarrage
X_scoring = pd.read_csv("data/processed/X_scoring.csv", index_col="SK_ID_CURR")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        approved, proba_default, threshold = predict_from_features(payload.features)
        row = pd.DataFrame([payload.features])
        shap_local = get_shap_local(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return PredictResponse(
        approved=approved,
        probability=proba_default,
        threshold=threshold,
        shap_local=shap_local,
        shap_global=SHAP_GLOBAL
    )
    
@app.get("/predict/{client_id}", response_model=PredictResponse)
def predict_by_id(client_id: int):
    if client_id not in X_scoring.index:
        raise HTTPException(status_code=404, detail="Client introuvable")

    row = X_scoring.loc[[client_id]]
    features = X_scoring.loc[client_id].to_dict()
    approved, proba, threshold = predict_from_features(features)
    shap_local = get_shap_local(row)

    return PredictResponse(
        approved=approved,
        probability=proba,
        threshold=threshold,
        shap_local=shap_local,
        shap_global=SHAP_GLOBAL
    )