import time
from typing import Any, Dict, List
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """Model inference with caching and latency tracking."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._load_times: Dict[str, float] = {}
        self._prediction_counts: Dict[str, int] = {}
        self._total_latencies: Dict[str, float] = {}

    def load_model(self, model_name: str, tracking_uri: str = None) -> bool:
        try:
            from registry.mlflow_registry import MLflowModelRegistry

            reg = MLflowModelRegistry(tracking_uri)
            model = reg.get_production_model(model_name)
            self._models[model_name] = model
            self._load_times[model_name] = time.time()
            self._prediction_counts[model_name] = 0
            self._total_latencies[model_name] = 0.0
            logger.info(f"Loaded production model: {model_name}")
            return True
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}")
            return False

    def predict(
        self,
        model_name: str,
        features: Dict[str, Any],
        return_proba: bool = False,
    ) -> Dict[str, Any]:
        if model_name not in self._models:
            self.load_model(model_name)
        model = self._models.get(model_name)
        if model is None:
            raise ValueError(f"Model not available: {model_name}")

        t0 = time.time()
        df = pd.DataFrame([features])
        if return_proba and hasattr(model, "predict_proba"):
            prediction = model.predict_proba(df)[0].tolist()
            label = model.predict(df)[0]
        else:
            prediction = model.predict(df)[0]
            label = prediction

        latency = (time.time() - t0) * 1000
        self._prediction_counts[model_name] = self._prediction_counts.get(model_name, 0) + 1
        self._total_latencies[model_name] = self._total_latencies.get(model_name, 0.0) + latency

        return {
            "prediction": label if not return_proba else prediction,
            "probability": prediction if return_proba else None,
            "latency_ms": latency,
        }

    def get_mean_latency_ms(self, model_name: str) -> float:
        count = self._prediction_counts.get(model_name, 0)
        total = self._total_latencies.get(model_name, 0.0)
        return total / count if count > 0 else 0.0

    def loaded_models(self) -> List[str]:
        return list(self._models.keys())
