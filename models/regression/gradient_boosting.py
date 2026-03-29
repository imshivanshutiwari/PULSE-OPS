from typing import Any, Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostRegressor(BaseModel):
    def __init__(self):
        super().__init__("xgboost_regressor", "regression")

    def build(self, params: Dict[str, Any]) -> None:
        self.model = xgb.XGBRegressor(
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 300),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]:
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.is_trained = True
        preds = self.model.predict(X_train)
        metrics = {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, preds))),
            "train_mae": float(mean_absolute_error(y_train, preds)),
            "train_r2": float(r2_score(y_train, preds)),
        }
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            metrics["val_mae"] = float(mean_absolute_error(y_val, val_preds))
            metrics["val_r2"] = float(r2_score(y_val, val_preds))
        return metrics

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(X).reshape(-1, 1)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        return {f"f{i}": float(s) for i, s in enumerate(self.model.feature_importances_)}


class LightGBMRegressor(BaseModel):
    def __init__(self):
        super().__init__("lightgbm_regressor", "regression")

    def build(self, params: Dict[str, Any]) -> None:
        self.model = lgb.LGBMRegressor(
            num_leaves=params.get("num_leaves", 31),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 500),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]:
        callbacks = [lgb.log_evaluation(period=-1)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(50, verbose=False))
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        self.is_trained = True
        preds = self.model.predict(X_train)
        metrics = {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, preds))),
            "train_r2": float(r2_score(y_train, preds)),
        }
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            metrics["val_r2"] = float(r2_score(y_val, val_preds))
        return metrics

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(X).reshape(-1, 1)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        return {f"f{i}": float(s) for i, s in enumerate(self.model.feature_importances_)}
