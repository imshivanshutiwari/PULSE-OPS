import pandas as pd
from drift.evidently_detector import EvidentlyDriftDetector, DriftReport
from utils.logger import get_logger

logger = get_logger(__name__)


class DataDriftAnalyzer:
    def __init__(self):
        self.detector = EvidentlyDriftDetector()

    def analyze(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        report = self.detector.detect_data_drift(reference, current)
        logger.info(
            f"Data drift: {report.dataset_drift}, share={report.drift_share:.3f}, "
            f"drifted cols={report.n_drifted_columns}"
        )
        return report

    def get_most_drifted_features(self, report: DriftReport, top_n: int = 5) -> list:
        drifted = [col for col, drifted in report.per_column_drift.items() if drifted]
        return drifted[:top_n]
