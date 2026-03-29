from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Registered model name")
    features: Dict[str, Any] = Field(..., description="Feature key-value pairs")
    return_proba: bool = Field(default=False)


class PredictionResponse(BaseModel):
    model_name: str
    prediction: Any
    probability: Optional[List[float]] = None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    stage: str
    version: str
    metrics: Dict[str, float]
