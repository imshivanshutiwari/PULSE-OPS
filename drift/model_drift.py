import pandas as pd
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from drift.evidently_detector import EvidentlyDriftDetector, PerformanceReport
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelDriftAnalyzer:
    def __init__(self):
        self.detector = EvidentlyDriftDetector()

    def analyze(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        target_col: str,
        prediction_col: str = None,
    ) -> PerformanceReport:
        mapping = ColumnMapping(target=target_col, prediction=prediction_col)
        report = self.detector.detect_model_performance(reference, current, mapping)
        logger.info(f"Model performance: acc={report.accuracy:.4f}, f1={report.f1:.4f}")
        return report

    def compute_performance_drop(
        self, reference_report: PerformanceReport, current_report: PerformanceReport
    ) -> float:
        return reference_report.accuracy - current_report.accuracy
