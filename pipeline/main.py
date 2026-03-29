"""PULSE-OPS end-to-end pipeline entry point.

Imports from ALL modules: data/, models/, drift/, retraining/,
orchestration/, registry/, monitoring/, feature_store/, dashboard/
"""

import argparse
from utils.logger import get_logger
from utils.config_loader import get_pipeline_config

logger = get_logger(__name__)


def run_full_pipeline(
    datasets: list = None,
    run_hpo: bool = False,
    skip_fetch: bool = False,
):
    """Run the complete PULSE-OPS end-to-end pipeline."""
    cfg = get_pipeline_config()
    datasets = datasets or cfg["datasets"]
    logger.info("=" * 60)
    logger.info("PULSE-OPS — FULL PIPELINE START")
    logger.info("=" * 60)

    # 1. DATA INGESTION
    if not skip_fetch:
        logger.info("Step 1: Data ingestion")
        from data.processors.dataset_builder import DatasetBuilder

        builder = DatasetBuilder()
        for dataset in datasets:
            try:
                if dataset == "adult":
                    builder.build_adult()
                elif dataset == "wine_quality":
                    builder.build_wine()
                elif dataset == "bike_sharing":
                    builder.build_bike()
                elif dataset == "german_credit":
                    builder.build_credit()
                logger.info(f"  ✓ {dataset}")
            except Exception as e:
                logger.warning(f"  ✗ {dataset}: {e}")

    # 2. FEATURE STORE MATERIALIZATION
    logger.info("Step 2: Feature store materialization")
    from feature_store.feature_pipeline import FeaturePipeline

    fp = FeaturePipeline()
    for dataset in datasets:
        try:
            count = fp.materialize_to_online(dataset)
            logger.info(f"  ✓ {dataset}: {count} records")
        except Exception as e:
            logger.warning(f"  ✗ {dataset}: {e}")

    # 3. MODEL TRAINING
    logger.info("Step 3: Model training")
    from models.trainer import ModelTrainer
    from models.classification.gradient_boosting import XGBoostClassifier, LightGBMClassifier
    from models.regression.gradient_boosting import XGBoostRegressor, LightGBMRegressor
    from data.processors.data_splitter import DataSplitter
    import pandas as pd

    model_map = cfg["active_models"]
    ModelClass_map = {
        "xgboost_classifier": XGBoostClassifier,
        "xgboost_regressor": XGBoostRegressor,
        "lightgbm_regressor": LightGBMRegressor,
        "lightgbm_classifier": LightGBMClassifier,
    }
    target_map = {
        "adult": "income",
        "wine_quality": "quality",
        "bike_sharing": "cnt",
        "german_credit": "creditworthiness",
    }

    training_results = {}
    for dataset in datasets:
        model_type = model_map.get(dataset)
        if model_type is None:
            continue
        try:
            X, y = fp.get_training_dataset(dataset, target_map[dataset])
            X_filled = X.fillna(0)
            splitter = DataSplitter()
            X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
                pd.concat([X_filled, y], axis=1), target_map[dataset]
            )
            ModelClass = ModelClass_map.get(model_type, XGBoostClassifier)
            model = ModelClass()
            trainer = ModelTrainer()
            result = trainer.train(model, X_train, y_train, X_val, y_val)
            training_results[dataset] = result
            logger.info(f"  ✓ {dataset}/{model_type}: run_id={result['run_id']}")
        except Exception as e:
            logger.warning(f"  ✗ {dataset}: {e}")

    # 4. MODEL REGISTRY
    logger.info("Step 4: Model registry")
    from registry.mlflow_registry import MLflowModelRegistry

    registry = MLflowModelRegistry()
    for dataset, result in training_results.items():
        model_type = model_map.get(dataset, "")
        try:
            version = registry.register_model(result["run_id"], model_type)
            registry.promote_to_staging(model_type, int(version.version))
            logger.info(f"  ✓ {model_type} v{version.version} → Staging")
        except Exception as e:
            logger.warning(f"  ✗ registry {dataset}: {e}")

    # 5. DRIFT DETECTION
    logger.info("Step 5: Drift detection")
    from drift.evidently_detector import EvidentlyDriftDetector
    from drift.drift_reporter import DriftReporter
    from retraining.trigger import DriftBasedRetrigger

    detector = EvidentlyDriftDetector()
    reporter = DriftReporter()
    trigger = DriftBasedRetrigger()

    for dataset in datasets:
        try:
            df = fp.compute_features(dataset)
            n = len(df)
            if n < 100:
                continue
            num_df = df.select_dtypes(include=["number"])
            reference = num_df.iloc[: n // 2]
            current = num_df.iloc[n // 2 :]
            common_cols = list(set(reference.columns) & set(current.columns))
            suite = detector.run_full_suite(reference[common_cols], current[common_cols])
            reporter.save_json_report(suite)
            needs_retrain = trigger.should_retrain(suite)
            logger.info(
                f"  ✓ {dataset}: drift={suite.data_drift.drift_share:.3f}, retrain={needs_retrain}"
            )
        except Exception as e:
            logger.warning(f"  ✗ drift {dataset}: {e}")

    # 6. MONITORING
    logger.info("Step 6: Monitoring setup")
    from monitoring.prometheus_exporter import PulseOpsPrometheusExporter
    from monitoring.health_checker import HealthChecker

    exporter = PulseOpsPrometheusExporter()
    try:
        exporter.start_server()
        logger.info("  ✓ Prometheus exporter started on port 8001")
    except Exception as e:
        logger.warning(f"  ✗ Prometheus: {e} — check if port 8001 is already in use")

    health = HealthChecker()
    checks = health.check_all()
    grade = health.compute_health_score(checks)
    logger.info(f"  System health grade: {grade}")

    logger.info("=" * 60)
    logger.info("PULSE-OPS — PIPELINE COMPLETE")
    logger.info("=" * 60)
    return training_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PULSE-OPS Pipeline")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to process")
    parser.add_argument("--hpo", action="store_true", help="Run HPO")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching")
    args = parser.parse_args()
    run_full_pipeline(
        datasets=args.datasets,
        run_hpo=args.hpo,
        skip_fetch=args.skip_fetch,
    )
