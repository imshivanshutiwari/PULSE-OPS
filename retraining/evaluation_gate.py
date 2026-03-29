from typing import Dict
from utils.config_loader import get_model_config
from utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationGate:
    """Performance gate check before model promotion."""

    def __init__(self):
        cfg = get_model_config()
        self.threshold = float(cfg.get("eval_metric_threshold", 0.02))
        self.min_accuracy = 0.70

    def check(
        self,
        new_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        metric_key: str = "val_accuracy",
    ) -> bool:
        new_metric = new_metrics.get(metric_key, new_metrics.get("val_r2", 0.0))
        current_metric = current_metrics.get(metric_key, current_metrics.get("val_r2", 0.0))

        if new_metric < self.min_accuracy and metric_key == "val_accuracy":
            logger.warning(f"Gate failed: {metric_key}={new_metric:.4f} < min {self.min_accuracy}")
            return False

        improvement = new_metric - current_metric
        if improvement > -self.threshold:
            logger.info(
                f"Gate passed: {metric_key} {current_metric:.4f} → {new_metric:.4f} "
                f"(improvement={improvement:+.4f})"
            )
            return True
        logger.warning(
            f"Gate failed: {metric_key} {current_metric:.4f} → {new_metric:.4f} "
            f"(degradation={improvement:.4f} > threshold={self.threshold})"
        )
        return False
