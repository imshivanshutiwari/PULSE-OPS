import pandas as pd
from drift.evidently_detector import EvidentlyDriftDetector, QualityReport
from utils.logger import get_logger

logger = get_logger(__name__)


class DataQualityAnalyzer:
    def __init__(self):
        self.detector = EvidentlyDriftDetector()

    def analyze(self, data: pd.DataFrame) -> QualityReport:
        report = self.detector.detect_data_quality(data)
        logger.info(
            f"Quality: missing={report.missing_values}, dups={report.duplicates}"
        )
        return report

    def compute_quality_score(self, report: QualityReport, n_rows: int) -> float:
        missing_penalty = sum(report.missing_values.values()) / max(n_rows, 1)
        dup_penalty = report.duplicates / max(n_rows, 1)
        score = max(0.0, 1.0 - missing_penalty - dup_penalty)
        return float(score)
