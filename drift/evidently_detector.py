from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.legacy.pipeline.column_mapping import ColumnMapping

# ModelPerformancePreset is not always available; provide a fallback
try:
    from evidently.legacy.metric_preset import ModelPerformancePreset
except ImportError:  # pragma: no cover
    ModelPerformancePreset = None  # type: ignore[assignment,misc]
from utils.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


@dataclass
class DriftReport:
    dataset_drift: bool
    drift_share: float
    n_drifted_columns: int
    per_column_drift: Dict[str, Any]


@dataclass
class PerformanceReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    per_class_metrics: Dict[str, Any]


@dataclass
class TargetDriftReport:
    target_drift_detected: bool
    p_value: float
    stattest_name: str
    target_correlations: Dict[str, float]


@dataclass
class QualityReport:
    missing_values: Dict[str, float]
    duplicates: int
    outliers: Dict[str, int]
    correlations: Dict[str, float]


@dataclass
class FullDriftSuite:
    data_drift: DriftReport
    model_perf: Optional[PerformanceReport]
    target_drift: Optional[TargetDriftReport]
    data_quality: QualityReport
    html_path: Optional[str] = None


class EvidentlyDriftDetector:
    """Full Evidently AI drift detection suite."""

    def detect_data_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result = report.as_dict()
        metrics = result.get("metrics", [{}])
        drift_result = {}
        for m in metrics:
            if m.get("metric") == "DatasetDriftMetric":
                drift_result = m.get("result", {})
                break
        if not drift_result:
            drift_result = metrics[0].get("result", {}) if metrics else {}

        per_col = {}
        for m in metrics:
            if m.get("metric") == "DataDriftTable":
                drift_by_col = m.get("result", {}).get("drift_by_columns", {})
                for col, val in drift_by_col.items():
                    per_col[col] = (
                        val.get("drift_detected", False) if isinstance(val, dict) else bool(val)
                    )

        return DriftReport(
            dataset_drift=bool(drift_result.get("dataset_drift", False)),
            drift_share=float(drift_result.get("share_of_drifted_columns", 0.0)),
            n_drifted_columns=int(drift_result.get("number_of_drifted_columns", 0)),
            per_column_drift=per_col,
        )

    def detect_model_performance(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: ColumnMapping,
    ) -> PerformanceReport:
        if ModelPerformancePreset is None:
            logger.warning("ModelPerformancePreset not available in this Evidently version")
            return PerformanceReport(accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.0, per_class_metrics={})
        report = Report(metrics=[ModelPerformancePreset()])
        try:
            report.run(
                reference_data=reference,
                current_data=current,
                column_mapping=column_mapping,
            )
            result = report.as_dict()
            metrics_list = result.get("metrics", [])
            perf = {}
            for m in metrics_list:
                r = m.get("result", {})
                if "current" in r:
                    perf = r["current"]
                    break
        except Exception as e:
            logger.warning(f"Model performance report error: {e}")
            perf = {}

        return PerformanceReport(
            accuracy=float(perf.get("accuracy", 0.0)),
            precision=float(perf.get("precision", 0.0)),
            recall=float(perf.get("recall", 0.0)),
            f1=float(perf.get("f1", 0.0)),
            roc_auc=float(perf.get("roc_auc", 0.0)),
            per_class_metrics=perf.get("per_class_metrics", {}),
        )

    def detect_target_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: ColumnMapping,
    ) -> TargetDriftReport:
        report = Report(metrics=[TargetDriftPreset()])
        try:
            report.run(
                reference_data=reference,
                current_data=current,
                column_mapping=column_mapping,
            )
            result = report.as_dict()
            metrics_list = result.get("metrics", [])
            target_result = {}
            for m in metrics_list:
                if "drift_detected" in m.get("result", {}):
                    target_result = m["result"]
                    break
        except Exception as e:
            logger.warning(f"Target drift report error: {e}")
            target_result = {}

        return TargetDriftReport(
            target_drift_detected=bool(target_result.get("drift_detected", False)),
            p_value=float(target_result.get("p_value", 1.0)),
            stattest_name=str(target_result.get("stattest_name", "unknown")),
            target_correlations={},
        )

    def detect_data_quality(self, data: pd.DataFrame) -> QualityReport:
        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=None, current_data=data)
        result = report.as_dict()
        metrics_list = result.get("metrics", [])
        missing = {}
        duplicates = 0
        outliers = {}
        for m in metrics_list:
            r = m.get("result", {})
            if "current" in r:
                curr = r["current"]
                if "number_of_missing_values" in curr:
                    missing["total"] = curr["number_of_missing_values"]
                if "number_of_duplicated_rows" in curr:
                    duplicates = curr["number_of_duplicated_rows"]
        return QualityReport(
            missing_values=missing,
            duplicates=int(duplicates),
            outliers=outliers,
            correlations={},
        )

    def run_full_suite(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: ColumnMapping = None,
    ) -> FullDriftSuite:
        if column_mapping is None:
            column_mapping = ColumnMapping()

        data_drift = self.detect_data_drift(reference, current)
        try:
            model_perf = self.detect_model_performance(reference, current, column_mapping)
        except Exception as e:
            logger.warning(f"Skipping model perf: {e}")
            model_perf = None
        try:
            target_drift = self.detect_target_drift(reference, current, column_mapping)
        except Exception as e:
            logger.warning(f"Skipping target drift: {e}")
            target_drift = None
        data_quality = self.detect_data_quality(current)

        suite = FullDriftSuite(
            data_drift=data_drift,
            model_perf=model_perf,
            target_drift=target_drift,
            data_quality=data_quality,
        )
        html_path = self.generate_html_report(suite, str(REPORTS_DIR / "full_drift_report.html"))
        suite.html_path = html_path
        return suite

    def generate_html_report(self, full_suite: FullDriftSuite, save_path: str) -> str:
        import json

        data = {
            "data_drift": {
                "dataset_drift": full_suite.data_drift.dataset_drift,
                "drift_share": full_suite.data_drift.drift_share,
                "n_drifted_columns": full_suite.data_drift.n_drifted_columns,
            },
            "data_quality": {
                "missing_values": full_suite.data_quality.missing_values,
                "duplicates": full_suite.data_quality.duplicates,
            },
        }
        html = f"""<!DOCTYPE html>
<html>
<head><title>PULSE-OPS Drift Report</title>
<style>body{{background:#050a06;color:#c0dd97;font-family:'JetBrains Mono',monospace;padding:20px;}}
h1{{color:#639922;}}h2{{color:#97c459;}}
.metric{{background:#0a1008;border:1px solid #0f1e0a;padding:10px;margin:5px;display:inline-block;}}
.drift{{color:#e24b4a;}}.ok{{color:#97c459;}}</style></head>
<body>
<h1>◈ PULSE-OPS Drift Report</h1>
<h2>Data Drift</h2>
<div class="metric">Dataset Drift: <span class="{'drift' if data['data_drift']['dataset_drift'] else 'ok'}">
{data['data_drift']['dataset_drift']}</span></div>
<div class="metric">Drift Share: {data['data_drift']['drift_share']:.3f}</div>
<div class="metric">Drifted Columns: {data['data_drift']['n_drifted_columns']}</div>
<h2>Data Quality</h2>
<div class="metric">Duplicates: {data['data_quality']['duplicates']}</div>
<pre>{json.dumps(data, indent=2)}</pre>
</body></html>"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved: {save_path}")
        return save_path
