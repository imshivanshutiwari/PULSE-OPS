import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from drift.evidently_detector import FullDriftSuite
from utils.logger import get_logger

logger = get_logger(__name__)
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


class DriftReporter:
    def save_json_report(self, suite: FullDriftSuite, path: str = None) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = path or str(REPORTS_DIR / f"drift_report_{timestamp}.json")
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_drift": {
                "dataset_drift": suite.data_drift.dataset_drift,
                "drift_share": suite.data_drift.drift_share,
                "n_drifted_columns": suite.data_drift.n_drifted_columns,
            },
            "data_quality": {
                "missing_values": suite.data_quality.missing_values,
                "duplicates": suite.data_quality.duplicates,
            },
        }
        if suite.model_perf:
            data["model_performance"] = {
                "accuracy": suite.model_perf.accuracy,
                "f1": suite.model_perf.f1,
            }
        if suite.target_drift:
            data["target_drift"] = {
                "detected": suite.target_drift.target_drift_detected,
                "p_value": suite.target_drift.p_value,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON report saved: {path}")
        return path

    def load_latest_report(self) -> Optional[dict]:
        reports = sorted(REPORTS_DIR.glob("drift_report_*.json"), reverse=True)
        if not reports:
            return None
        with open(reports[0]) as f:
            return json.load(f)
