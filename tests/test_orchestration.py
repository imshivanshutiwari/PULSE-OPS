"""4 orchestration flows tests."""


def test_training_flow_returns_run_id(small_adult_df):
    """training_flow should complete and return a run_id."""
    from data.processors.feature_engineer import FeatureEngineer
    from pathlib import Path

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    parquet_path = raw_dir / "adult_features.parquet"
    if not parquet_path.exists():
        engineer = FeatureEngineer()
        df = engineer.handle_missing(small_adult_df.copy())
        df = engineer.encode_categoricals(df)
        df = engineer.add_event_timestamp(df)
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

    from orchestration.training_flow import training_flow

    result = training_flow(dataset_name="adult", model_type="xgboost_classifier", run_hpo=False)
    assert "run_id" in result
    assert result["run_id"] is not None


def test_drift_flow_creates_report(small_adult_df):
    """drift_detection_flow should complete and return drift info."""
    from data.processors.feature_engineer import FeatureEngineer
    from pathlib import Path

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    parquet_path = raw_dir / "adult_features.parquet"
    if not parquet_path.exists():
        engineer = FeatureEngineer()
        df = engineer.handle_missing(small_adult_df.copy())
        df = engineer.encode_categoricals(df)
        df = engineer.add_event_timestamp(df)
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

    from orchestration.drift_flow import drift_detection_flow

    result = drift_detection_flow(dataset_name="adult")
    assert "drift_detected" in result
    assert "report_path" in result


def test_data_validator_passes_clean_data(small_adult_df):
    """Data validator should pass on clean dataset."""
    from data.processors.data_validator import DataValidator
    from data.processors.feature_engineer import FeatureEngineer

    engineer = FeatureEngineer()
    df = engineer.handle_missing(small_adult_df.copy())
    validator = DataValidator()
    result = validator.validate(df)
    assert "missing_check" in result
    assert result["missing_check"]["passed"] is True


def test_prefect_flows_importable():
    """All Prefect flows should be importable."""
    from orchestration.prefect_flows import (
        training_flow,
        drift_detection_flow,
        retraining_flow,
        deployment_flow,
    )

    assert callable(training_flow)
    assert callable(drift_detection_flow)
    assert callable(retraining_flow)
    assert callable(deployment_flow)
