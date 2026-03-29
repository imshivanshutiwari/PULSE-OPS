from typing import Optional, List
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from utils.config_loader import get_pipeline_config
from utils.logger import get_logger

logger = get_logger(__name__)


class MLflowModelRegistry:
    """MLflow Model Registry with full stage lifecycle management."""

    def __init__(self, tracking_uri: str = None):
        cfg = get_pipeline_config()
        self.tracking_uri = tracking_uri or cfg["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
    ) -> ModelVersion:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered {model_name} v{version.version} from run {run_id}")
        return version

    def promote_to_staging(self, model_name: str, version: int) -> ModelVersion:
        result = self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info(f"Promoted {model_name} v{version} → Staging")
        return result

    def promote_to_production(
        self,
        model_name: str,
        version: int,
        champion_version: Optional[int] = None,
        metric_threshold: float = 0.01,
    ) -> bool:
        if champion_version is not None:
            try:
                challenger_run = self._get_run_for_version(model_name, str(version))
                champion_run = self._get_run_for_version(model_name, str(champion_version))
                challenger_metric = self._best_metric(challenger_run)
                champion_metric = self._best_metric(champion_run)
                if challenger_metric <= champion_metric + metric_threshold:
                    logger.info(
                        f"Challenger ({challenger_metric:.4f}) not better than "
                        f"champion ({champion_metric:.4f}), keeping champion"
                    )
                    return False
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=str(champion_version),
                    stage="Archived",
                )
            except Exception as e:
                logger.warning(f"A/B comparison failed: {e}")

        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Promoted {model_name} v{version} → Production")
        return True

    def rollback(self, model_name: str) -> Optional[ModelVersion]:
        versions = self.client.search_model_versions(f"name='{model_name}'")
        archived = [
            v for v in versions
            if v.current_stage == "Archived"
        ]
        if not archived:
            logger.warning(f"No archived versions for {model_name}")
            return None
        archived.sort(key=lambda v: int(v.version), reverse=True)
        last_archived = archived[0]
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        for v in prod_versions:
            self.client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Archived"
            )
        result = self.client.transition_model_version_stage(
            name=model_name,
            version=last_archived.version,
            stage="Production",
        )
        logger.info(f"Rolled back {model_name} to v{last_archived.version}")
        return result

    def get_production_model(self, model_name: str):
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pyfunc.load_model(model_uri)

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        return self.client.search_model_versions(f"name='{model_name}'")

    def _get_run_for_version(self, model_name: str, version: str):
        mv = self.client.get_model_version(model_name, version)
        return self.client.get_run(mv.run_id)

    def _best_metric(self, run) -> float:
        metrics = run.data.metrics
        for key in ["val_accuracy", "val_f1", "val_r2", "accuracy"]:
            if key in metrics:
                return metrics[key]
        if metrics:
            return list(metrics.values())[0]
        return 0.0
