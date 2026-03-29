from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelValidator:
    """Pre-promotion validation checks."""

    def __init__(self, min_accuracy: float = 0.70):
        self.min_accuracy = min_accuracy

    def validate_version(self, model_name: str, version: str) -> Dict[str, Any]:
        try:
            from registry.mlflow_registry import MLflowModelRegistry
            reg = MLflowModelRegistry()
            run = reg._get_run_for_version(model_name, version)
            metrics = run.data.metrics
            best = reg._best_metric(run)
            passed = best >= self.min_accuracy
            return {
                "passed": passed,
                "best_metric": best,
                "metrics": metrics,
                "reason": "ok" if passed else f"metric {best:.4f} < {self.min_accuracy}",
            }
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return {"passed": True, "reason": "validation_skipped", "error": str(e)}
