from fastapi import FastAPI, HTTPException
from api.app.schemas import PredictRequest, PredictResponse
from api.app.model_service import predict_from_features

app = FastAPI(title="OC Projet 7 - API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse) # response_model = format de la reponse (défini dans Schemas.py)
def predict(payload: PredictRequest): # payload = format du payload de la requete
    try:
        approved, proba_default, threshold = predict_from_features(payload.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return PredictResponse(
        approved=approved,
        probability=proba_default,
        threshold=threshold
    )
    
    