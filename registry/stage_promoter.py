from registry.mlflow_registry import MLflowModelRegistry, ALIAS_PRODUCTION, ALIAS_STAGING
from registry.model_validator import ModelValidator
from utils.logger import get_logger

logger = get_logger(__name__)


class StagePromoter:
    """Manages model lifecycle transitions with validation gates (MLflow 3.x alias-based)."""

    def __init__(self):
        self.registry = MLflowModelRegistry()
        self.validator = ModelValidator()

    def promote_staging_to_production(
        self, model_name: str, challenger_version: int, champion_version: int = None
    ) -> bool:
        validation = self.validator.validate_version(model_name, str(challenger_version))
        if not validation["passed"]:
            logger.warning(
                f"Validation failed for {model_name} v{challenger_version}: {validation}"
            )
            return False
        result = self.registry.promote_to_production(
            model_name, challenger_version, champion_version
        )
        return result

    def archive_old_versions(self, model_name: str, keep_n: int = 3) -> int:
        versions = self.registry.list_versions(model_name)
        # Versions that hold neither 'production' nor 'staging' alias
        unaliased = sorted(
            [
                v
                for v in versions
                if ALIAS_PRODUCTION not in (v.aliases or [])
                and ALIAS_STAGING not in (v.aliases or [])
            ],
            key=lambda v: int(v.version),
        )
        n_deleted = 0
        for v in unaliased[:-keep_n] if len(unaliased) > keep_n else []:
            try:
                self.registry.client.delete_model_version(model_name, v.version)
                n_deleted += 1
            except Exception as e:
                logger.warning(f"Could not delete version {v.version}: {e}")
        return n_deleted
