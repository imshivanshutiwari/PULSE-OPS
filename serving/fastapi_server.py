import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from serving.predictor import ModelPredictor
from serving.schemas import PredictionRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from utils.logger import get_logger

logger = get_logger(__name__)

_predictor = ModelPredictor()
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI server starting")
    yield
    logger.info("FastAPI server stopping")


app = FastAPI(
    title="PULSE-OPS Model Serving",
    description="Real-time ML model inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        models_loaded=_predictor.loaded_models(),
        uptime_seconds=time.time() - _start_time,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = _predictor.predict(
            request.model_name,
            request.features,
            return_proba=request.return_proba,
        )
        return PredictionResponse(
            model_name=request.model_name,
            prediction=result["prediction"],
            probability=result.get("probability"),
            latency_ms=result["latency_ms"],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[str])
async def list_models():
    return _predictor.loaded_models()


@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    success = _predictor.load_model(model_name)
    return {"success": success, "model_name": model_name}
