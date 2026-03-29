from typing import Optional
from drift.evidently_detector import FullDriftSuite
from utils.config_loader import get_drift_config
from utils.logger import get_logger

logger = get_logger(__name__)


class DriftBasedRetrigger:
    """Automatic retraining trigger based on drift detection."""

    def __init__(self):
        cfg = get_drift_config()
        self.data_threshold = float(cfg.get("data_drift_threshold", 0.3))
        self.perf_threshold = float(cfg.get("performance_drop_threshold", 0.05))
        self.target_alpha = float(cfg.get("target_drift_alpha", 0.05))
        self.quality_threshold = float(cfg.get("data_quality_threshold", 0.9))

    def should_retrain(self, drift_suite: FullDriftSuite) -> bool:
        reasons = []
        if drift_suite.data_drift.drift_share > self.data_threshold:
            reasons.append(
                f"data_drift_share={drift_suite.data_drift.drift_share:.3f} > {self.data_threshold}"
            )
        if drift_suite.model_perf and drift_suite.model_perf.accuracy < (1.0 - self.perf_threshold):
            reasons.append(
                f"model_accuracy_drop detected (acc={drift_suite.model_perf.accuracy:.3f})"
            )
        if (
            drift_suite.target_drift
            and drift_suite.target_drift.target_drift_detected
            and drift_suite.target_drift.p_value < self.target_alpha
        ):
            reasons.append(
                f"target_drift p_value={drift_suite.target_drift.p_value:.4f} < {self.target_alpha}"
            )
        quality_score = self._compute_quality_score(drift_suite)
        if quality_score < self.quality_threshold:
            reasons.append(f"data_quality_score={quality_score:.3f} < {self.quality_threshold}")
        if reasons:
            for r in reasons:
                logger.warning(f"Retrain trigger: {r}")
            return True
        logger.info("No retraining needed")
        return False

    def _compute_quality_score(self, drift_suite: FullDriftSuite) -> float:
        missing = sum(drift_suite.data_quality.missing_values.values())
        dups = drift_suite.data_quality.duplicates
        total = missing + dups
        return max(0.0, 1.0 - (total / 10000))

    def compute_urgency(self, drift_suite: FullDriftSuite) -> str:
        share = drift_suite.data_drift.drift_share
        if share > 0.5:
            return "CRITICAL"
        elif share > 0.3:
            return "HIGH"
        elif drift_suite.model_perf and drift_suite.model_perf.accuracy < 0.85:
            return "MEDIUM"
        else:
            return "LOW"

    def schedule_retraining(self, urgency: str) -> str:
        schedule_map = {
            "CRITICAL": "immediate",
            "HIGH": "next_hour",
            "MEDIUM": "next_day",
            "LOW": "weekly",
        }
        schedule = schedule_map.get(urgency, "next_day")
        logger.info(f"Retraining scheduled: {schedule} (urgency={urgency})")
        return schedule
