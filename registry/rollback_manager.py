from registry.mlflow_registry import MLflowModelRegistry, ALIAS_PRODUCTION
from utils.logger import get_logger

logger = get_logger(__name__)


class RollbackManager:
    """Auto-rollback on failure."""

    def __init__(self):
        self.registry = MLflowModelRegistry()

    def rollback_to_previous(self, model_name: str) -> bool:
        result = self.registry.rollback(model_name)
        return result is not None

    def health_check_and_rollback(self, model_name: str, min_accuracy: float = 0.75) -> bool:
        try:
            prod = self.registry._get_version_by_alias(model_name, ALIAS_PRODUCTION)
            if prod is None:
                logger.warning(f"No production version for {model_name}")
                return False
            run = self.registry.client.get_run(prod.run_id)
            best = self.registry._best_metric(run)
            if best < min_accuracy:
                logger.warning(
                    f"Production model below threshold ({best:.4f} < {min_accuracy}), rolling back"
                )
                return self.rollback_to_previous(model_name)
            logger.info(f"Production model healthy: {best:.4f}")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
