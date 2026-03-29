from typing import Any, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostClassifier(BaseModel):
    def __init__(self):
        super().__init__("xgboost_classifier", "classification")

    def build(self, params: Dict[str, Any]) -> None:
        self.model = xgb.XGBClassifier(
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 300),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]:
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_trained = True
        preds = self.model.predict(X_train)
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, preds)),
            "train_f1": float(f1_score(y_train, preds, average="weighted")),
        }
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            val_proba = self.model.predict_proba(X_val)
            metrics["val_accuracy"] = float(accuracy_score(y_val, val_preds))
            metrics["val_f1"] = float(f1_score(y_val, val_preds, average="weighted"))
            try:
                if val_proba.shape[1] == 2:
                    metrics["val_roc_auc"] = float(roc_auc_score(y_val, val_proba[:, 1]))
                else:
                    metrics["val_roc_auc"] = float(roc_auc_score(y_val, val_proba, multi_class="ovr"))
            except Exception:
                pass
        logger.info(f"XGBoost trained: {metrics}")
        return metrics

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        scores = self.model.feature_importances_
        return {f"f{i}": float(s) for i, s in enumerate(scores)}


class LightGBMClassifier(BaseModel):
    def __init__(self):
        super().__init__("lightgbm_classifier", "classification")

    def build(self, params: Dict[str, Any]) -> None:
        self.model = lgb.LGBMClassifier(
            num_leaves=params.get("num_leaves", 31),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 500),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]:
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if X_val is not None else [lgb.log_evaluation(period=-1)],
        )
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
        scores = self.model.feature_importances_
        return {f"f{i}": float(s) for i, s in enumerate(scores)}
