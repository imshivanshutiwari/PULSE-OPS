from registry.mlflow_registry import MLflowModelRegistry
from registry.model_validator import ModelValidator
from utils.logger import get_logger

logger = get_logger(__name__)


class StagePromoter:
    """Manages model stage transitions with validation gates."""

    def __init__(self):
        self.registry = MLflowModelRegistry()
        self.validator = ModelValidator()

    def promote_staging_to_production(
        self, model_name: str, challenger_version: int, champion_version: int = None
    ) -> bool:
        validation = self.validator.validate_version(model_name, str(challenger_version))
        if not validation["passed"]:
            logger.warning(f"Validation failed for {model_name} v{challenger_version}: {validation}")
            return False
        result = self.registry.promote_to_production(
            model_name, challenger_version, champion_version
        )
        return result

    def archive_old_versions(self, model_name: str, keep_n: int = 3) -> int:
        versions = self.registry.list_versions(model_name)
        archived = sorted(
            [v for v in versions if v.current_stage not in ("Production", "Staging")],
            key=lambda v: int(v.version),
        )
        n_archived = 0
        for v in archived[:-keep_n] if len(archived) > keep_n else []:
            try:
                self.registry.client.delete_model_version(model_name, v.version)
                n_archived += 1
            except Exception as e:
                logger.warning(f"Could not delete version {v.version}: {e}")
        return n_archived
