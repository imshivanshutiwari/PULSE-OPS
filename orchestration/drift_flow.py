from datetime import datetime
from pathlib import Path
from prefect import flow, task
from utils.logger import get_logger

logger = get_logger(__name__)
REPORTS_DIR = Path(__file__).parent.parent / "drift" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@task(name="load-reference-data", retries=2)
def load_reference_data_task(dataset_name: str):
    from feature_store.feature_pipeline import FeaturePipeline

    fp = FeaturePipeline()
    df = fp.compute_features(dataset_name)
    n = len(df)
    return df.iloc[: n // 2]


@task(name="load-current-data", retries=2)
def load_current_data_task(dataset_name: str):
    from feature_store.feature_pipeline import FeaturePipeline
    from data.fetchers.synthetic_drifter import PhysicsBasedDriftSimulator

    fp = FeaturePipeline()
    df = fp.compute_features(dataset_name)
    n = len(df)
    current = df.iloc[n // 2 :]
    return current


@task(name="run-evidently")
def run_evidently_task(reference, current):
    from drift.evidently_detector import EvidentlyDriftDetector
    from evidently.legacy.pipeline.column_mapping import ColumnMapping

    detector = EvidentlyDriftDetector()
    num_ref = reference.select_dtypes(include=["number"])
    num_cur = current.select_dtypes(include=["number"])
    if len(num_ref.columns) > 0 and len(num_cur.columns) > 0:
        common = list(set(num_ref.columns) & set(num_cur.columns))
        suite = detector.run_full_suite(num_ref[common], num_cur[common])
    else:
        suite = detector.run_full_suite(reference, current)
    return suite


@task(name="store-drift-report")
def store_drift_report_task(suite):
    from drift.drift_reporter import DriftReporter

    reporter = DriftReporter()
    path = reporter.save_json_report(suite)
    logger.info(f"Drift report stored: {path}")
    return path


@task(name="check-should-retrain")
def should_retrain_task(suite):
    from retraining.trigger import DriftBasedRetrigger

    trigger = DriftBasedRetrigger()
    return trigger.should_retrain(suite)


@flow(name="drift-detection-flow", description="Scheduled drift check with Evidently AI")
def drift_detection_flow(dataset_name: str = "adult"):
    logger.info(f"Drift flow: dataset={dataset_name}")
    reference = load_reference_data_task(dataset_name)
    current = load_current_data_task(dataset_name)
    suite = run_evidently_task(reference, current)
    report_path = store_drift_report_task(suite)
    needs_retrain = should_retrain_task(suite)
    if needs_retrain:
        logger.warning(f"Drift detected for {dataset_name}, retraining triggered")
    return {
        "dataset": dataset_name,
        "drift_detected": suite.data_drift.dataset_drift,
        "drift_share": suite.data_drift.drift_share,
        "report_path": report_path,
        "needs_retrain": needs_retrain,
    }


if __name__ == "__main__":
    from utils.config_loader import get_pipeline_config

    cfg = get_pipeline_config()
    for dataset in cfg["datasets"]:
        drift_detection_flow(dataset_name=dataset)
