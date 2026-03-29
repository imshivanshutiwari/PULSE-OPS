from typing import Dict, List
from utils.config_loader import get_monitoring_config
from utils.logger import get_logger

logger = get_logger(__name__)


class AlertRules:
    """Alertmanager-style alert rules for PULSE-OPS."""

    def __init__(self):
        cfg = get_monitoring_config()
        self.min_accuracy = float(cfg.get("alert_accuracy_threshold", 0.80))
        self.max_latency_ms = float(cfg.get("alert_latency_ms", 200))
        self.max_drift = float(cfg.get("alert_drift_threshold", 0.4))
        self._active_alerts: List[Dict] = []

    def check_accuracy(self, model_name: str, accuracy: float) -> bool:
        if accuracy < self.min_accuracy:
            alert = {
                "name": "ModelAccuracyLow",
                "severity": "warning",
                "model": model_name,
                "value": accuracy,
                "threshold": self.min_accuracy,
                "message": f"Model {model_name} accuracy {accuracy:.4f} below {self.min_accuracy}",
            }
            self._active_alerts.append(alert)
            logger.warning(alert["message"])
            return True
        return False

    def check_latency(self, model_name: str, latency_ms: float) -> bool:
        if latency_ms > self.max_latency_ms:
            alert = {
                "name": "ModelLatencyHigh",
                "severity": "critical",
                "model": model_name,
                "value": latency_ms,
                "threshold": self.max_latency_ms,
                "message": f"Model {model_name} latency {latency_ms:.1f}ms above {self.max_latency_ms}ms",
            }
            self._active_alerts.append(alert)
            logger.warning(alert["message"])
            return True
        return False

    def check_drift(self, dataset_name: str, drift_score: float) -> bool:
        if drift_score > self.max_drift:
            alert = {
                "name": "DataDriftHigh",
                "severity": "critical",
                "dataset": dataset_name,
                "value": drift_score,
                "threshold": self.max_drift,
                "message": f"Drift score {drift_score:.4f} above {self.max_drift} for {dataset_name}",
            }
            self._active_alerts.append(alert)
            logger.warning(alert["message"])
            return True
        return False

    def get_active_alerts(self) -> List[Dict]:
        return self._active_alerts

    def clear_resolved_alerts(self) -> int:
        cleared = len(self._active_alerts)
        self._active_alerts = []
        return cleared
