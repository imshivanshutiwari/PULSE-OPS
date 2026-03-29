from typing import Any, Dict, Optional
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from utils.logger import get_logger
from utils.config_loader import get_pipeline_config, get_model_config

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, tracking_uri: str = None):
        cfg = get_pipeline_config()
        self.tracking_uri = tracking_uri or cfg["mlflow"]["tracking_uri"]
        self.experiment_name = cfg["mlflow"]["experiment_name"]
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def train(
        self,
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        params: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        model_cfg = get_model_config()
        model_type = model.model_name
        default_params = model_cfg.get(model_type.replace("_classifier", "").replace("_regressor", ""), {})
        final_params = {**default_params, **(params or {})}
        model.build(final_params)

        with mlflow.start_run(run_name=f"{model_type}_run") as run:
            mlflow.log_params(final_params)
            if tags:
                mlflow.set_tags(tags)
            try:
                if "xgboost" in model_type:
                    mlflow.xgboost.autolog(log_input_examples=False, silent=True)
                elif "lightgbm" in model_type:
                    mlflow.lightgbm.autolog(log_input_examples=False, silent=True)
                else:
                    mlflow.sklearn.autolog(log_input_examples=False, silent=True)
            except Exception:
                pass

            metrics = model.fit(X_train, y_train, X_val, y_val)
            mlflow.log_metrics(metrics)

            artifact_path = "model"
            if hasattr(model.model, "save_model"):
                mlflow.xgboost.log_model(model.model, artifact_path)
            elif hasattr(model.model, "booster_"):
                mlflow.lightgbm.log_model(model.model, artifact_path)
            else:
                mlflow.sklearn.log_model(model.model, artifact_path)

            run_id = run.info.run_id
            logger.info(f"Run {run_id}: {metrics}")
            return {
                "run_id": run_id,
                "metrics": metrics,
                "artifact_path": artifact_path,
                "model_name": model_type,
            }
