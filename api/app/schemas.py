from pydantic import BaseModel
from typing import Any, Dict


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    approved: bool
    probability: float
    threshold: float
    shap_local: dict[str, float]
    shap_global: dict[str, float]