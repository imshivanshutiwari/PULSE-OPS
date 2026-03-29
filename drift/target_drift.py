import pandas as pd
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from drift.evidently_detector import EvidentlyDriftDetector, TargetDriftReport
from utils.logger import get_logger

logger = get_logger(__name__)


class TargetDriftAnalyzer:
    def __init__(self):
        self.detector = EvidentlyDriftDetector()

    def analyze(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        target_col: str,
    ) -> TargetDriftReport:
        mapping = ColumnMapping(target=target_col)
        report = self.detector.detect_target_drift(reference, current, mapping)
        logger.info(
            f"Target drift: detected={report.target_drift_detected}, p={report.p_value:.4f}"
        )
        return report
