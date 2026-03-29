from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.is_trained = False

    @abstractmethod
    def build(self, params: Dict[str, Any]) -> None:
        """Build/instantiate the model with given hyperparameters."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict[str, float]:
        """Train the model, return training metrics."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions (classifiers) or raw output (regressors)."""

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""

    def get_params(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}
