from typing import Optional, List
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from utils.config_loader import get_pipeline_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Alias constants — replaces the removed stage system in MLflow 3.x
ALIAS_STAGING = "staging"
ALIAS_PRODUCTION = "production"
ALIAS_ARCHIVED_PREFIX = "archived_v"


class MLflowModelRegistry:
    """MLflow Model Registry using alias-based lifecycle (MLflow 3.x compatible).

    Alias mapping that preserves stage semantics:
      Staging    → alias 'staging'
      Production → alias 'production'
      Archived   → alias 'archived_v{version}'
    """

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
        self.client.set_registered_model_alias(model_name, ALIAS_STAGING, str(version))
        mv = self.client.get_model_version(model_name, str(version))
        logger.info(f"Promoted {model_name} v{version} → staging (alias)")
        return mv

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
                # Archive the outgoing champion
                self._archive_version(model_name, champion_version)
            except Exception as e:
                logger.warning(f"A/B comparison failed: {e}")

        # Remove production alias from any existing production version before promoting
        self._clear_alias(model_name, ALIAS_PRODUCTION)
        self.client.set_registered_model_alias(model_name, ALIAS_PRODUCTION, str(version))
        logger.info(f"Promoted {model_name} v{version} → production (alias)")
        return True

    def rollback(self, model_name: str) -> Optional[ModelVersion]:
        """Rollback: demote current production alias, restore the most recent archived version."""
        versions = self.client.search_model_versions(f"name='{model_name}'")

        # Find current production version
        current_prod = self._get_version_by_alias(model_name, ALIAS_PRODUCTION)

        # Find archived versions (those with archived_v* aliases)
        archived = []
        for v in versions:
            for alias in (v.aliases or []):
                if alias.startswith(ALIAS_ARCHIVED_PREFIX):
                    archived.append(v)
                    break

        if not archived:
            logger.warning(f"No archived versions for {model_name}")
            return None

        archived.sort(key=lambda v: int(v.version), reverse=True)
        restore_version = archived[0]

        # Archive the current production version
        if current_prod is not None:
            self._archive_version(model_name, int(current_prod.version))

        # Restore archived version to production
        self._clear_alias(model_name, ALIAS_PRODUCTION)
        self.client.set_registered_model_alias(
            model_name, ALIAS_PRODUCTION, restore_version.version
        )
        # Remove its archived alias since it's now production
        self._clear_alias(
            model_name, f"{ALIAS_ARCHIVED_PREFIX}{restore_version.version}"
        )
        logger.info(f"Rolled back {model_name} to v{restore_version.version}")
        return self.client.get_model_version(model_name, restore_version.version)

    def get_production_model(self, model_name: str):
        model_uri = f"models:/{model_name}@{ALIAS_PRODUCTION}"
        return mlflow.pyfunc.load_model(model_uri)

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        return self.client.search_model_versions(f"name='{model_name}'")

    def get_version_stage(self, model_name: str, version: int) -> str:
        """Return a human-readable stage string based on the version's aliases."""
        try:
            mv = self.client.get_model_version(model_name, str(version))
            aliases = mv.aliases or []
            if ALIAS_PRODUCTION in aliases:
                return "Production"
            if ALIAS_STAGING in aliases:
                return "Staging"
            for a in aliases:
                if a.startswith(ALIAS_ARCHIVED_PREFIX):
                    return "Archived"
            return "None"
        except Exception:
            return "None"

    # ------------------------------------------------------------------ helpers

    def _get_version_by_alias(self, model_name: str, alias: str) -> Optional[ModelVersion]:
        try:
            return self.client.get_model_version_by_alias(model_name, alias)
        except Exception:
            return None

    def _archive_version(self, model_name: str, version: int) -> None:
        archive_alias = f"{ALIAS_ARCHIVED_PREFIX}{version}"
        self.client.set_registered_model_alias(model_name, archive_alias, str(version))
        self._clear_alias(model_name, ALIAS_PRODUCTION)

    def _clear_alias(self, model_name: str, alias: str) -> None:
        try:
            self.client.delete_registered_model_alias(model_name, alias)
        except Exception:
            pass  # alias may not exist yet

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
