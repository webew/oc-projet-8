from pydantic import BaseModel
from typing import Any, Dict


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    approved: bool
    probability: float
    threshold: float