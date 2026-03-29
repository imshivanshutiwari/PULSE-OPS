from typing import Dict, Any, Optional
import pandas as pd
from retraining.trigger import DriftBasedRetrigger
from retraining.hyperopt_tuner import OptunaHPOTuner
from retraining.evaluation_gate import EvaluationGate
from drift.evidently_detector import FullDriftSuite
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrainingPipeline:
    """Full retraining workflow with HPO, evaluation gate, and registry promotion."""

    def __init__(self):
        self.trigger = DriftBasedRetrigger()
        self.tuner = OptunaHPOTuner(n_trials=20)
        self.gate = EvaluationGate()

    def run(
        self,
        dataset_name: str,
        model_type: str,
        drift_suite: Optional[FullDriftSuite] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        if not force and drift_suite and not self.trigger.should_retrain(drift_suite):
            return {"status": "skipped", "reason": "no_drift_detected"}

        from feature_store.feature_pipeline import FeaturePipeline
        from data.processors.data_splitter import DataSplitter

        target_map = {
            "adult": "income",
            "wine_quality": "quality",
            "bike_sharing": "cnt",
            "german_credit": "creditworthiness",
        }
        target_col = target_map.get(dataset_name, "target")
        fp = FeaturePipeline()
        X, y = fp.get_training_dataset(dataset_name, target_col)
        X_filled = X.fillna(0)
        splitter = DataSplitter()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            pd.concat([X_filled, y], axis=1), target_col
        )

        task = "classification" if "classifier" in model_type else "regression"
        if "xgboost" in model_type:
            best_params = self.tuner.tune_xgboost(X_train, y_train, task=task)
        elif "lightgbm" in model_type:
            best_params = self.tuner.tune_lightgbm(X_train, y_train, task=task)
        else:
            best_params = self.tuner.tune_neural_net(X_train, y_train, task=task)

        from models.trainer import ModelTrainer
        from models.classification.gradient_boosting import XGBoostClassifier, LightGBMClassifier
        from models.regression.gradient_boosting import XGBoostRegressor, LightGBMRegressor

        model_map = {
            "xgboost_classifier": XGBoostClassifier,
            "xgboost_regressor": XGBoostRegressor,
            "lightgbm_regressor": LightGBMRegressor,
            "lightgbm_classifier": LightGBMClassifier,
        }
        ModelClass = model_map.get(model_type)
        if ModelClass is None:
            return {"status": "error", "reason": f"Unknown model type: {model_type}"}

        model = ModelClass()
        trainer = ModelTrainer()
        result = trainer.train(model, X_train, y_train, X_val, y_val, params=best_params)

        gate_passed = True
        if current_metrics:
            gate_passed = self.gate.check(result["metrics"], current_metrics)

        return {
            "status": "completed",
            "gate_passed": gate_passed,
            "run_id": result["run_id"],
            "metrics": result["metrics"],
            "best_params": best_params,
            "model_type": model_type,
            "dataset_name": dataset_name,
        }
