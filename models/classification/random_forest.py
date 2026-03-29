from typing import Any, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import accuracy_score, f1_score
from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestClassifier(BaseModel):
    def __init__(self):
        super().__init__("random_forest_classifier", "classification")

    def build(self, params: Dict[str, Any]) -> None:
        self.model = SklearnRF(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]:
        self.model.fit(X_train, y_train)
        self.is_trained = True
        preds = self.model.predict(X_train)
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, preds)),
            "train_f1": float(f1_score(y_train, preds, average="weighted")),
        }
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            metrics["val_accuracy"] = float(accuracy_score(y_val, val_preds))
            metrics["val_f1"] = float(f1_score(y_val, val_preds, average="weighted"))
        return metrics

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        return {f"f{i}": float(s) for i, s in enumerate(self.model.feature_importances_)}
