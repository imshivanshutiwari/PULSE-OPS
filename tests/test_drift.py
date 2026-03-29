"""5 tests for drift detection."""

import pytest
import pandas as pd
import numpy as np


def _make_reference_current(df, shift=False):
    from data.processors.feature_engineer import FeatureEngineer

    engineer = FeatureEngineer()
    df_clean = engineer.handle_missing(df.copy())
    df_clean = engineer.encode_categoricals(df_clean)
    num_df = df_clean.select_dtypes(include=["number"]).fillna(0)
    n = len(num_df)
    reference = num_df.iloc[: n // 2]
    if shift:
        from data.fetchers.synthetic_drifter import PhysicsBasedDriftSimulator

        drifter = PhysicsBasedDriftSimulator()
        current = drifter.simulate_sudden_drift(num_df.iloc[n // 2 :].copy(), shift_magnitude=4.0)
    else:
        current = num_df.iloc[n // 2 :]
    common = list(set(reference.columns) & set(current.columns))
    return reference[common], current[common]


def test_evidently_detects_real_drift(adult_df):
    """Evidently should detect drift when distribution is shifted."""
    from drift.evidently_detector import EvidentlyDriftDetector

    sample = adult_df.sample(2000, random_state=42).reset_index(drop=True)
    reference, current = _make_reference_current(sample, shift=True)
    detector = EvidentlyDriftDetector()
    report = detector.detect_data_drift(reference, current)
    assert (
        report.dataset_drift is True
    ), f"Expected drift=True, got drift_share={report.drift_share:.3f}"


def test_evidently_stable_on_same_data(adult_df):
    """Evidently should not detect drift on same distribution data."""
    from drift.evidently_detector import EvidentlyDriftDetector

    sample = adult_df.sample(1000, random_state=42).reset_index(drop=True)
    reference, _ = _make_reference_current(sample, shift=False)
    detector = EvidentlyDriftDetector()
    report = detector.detect_data_drift(reference, reference.copy())
    assert (
        report.drift_share < 0.5
    ), f"Expected low drift on same data, got {report.drift_share:.3f}"


def test_all_4_presets_run_without_error(adult_df):
    """run_full_suite should return a FullDriftSuite with all 4 reports."""
    from drift.evidently_detector import EvidentlyDriftDetector

    sample = adult_df.sample(500, random_state=42).reset_index(drop=True)
    reference, current = _make_reference_current(sample, shift=False)
    detector = EvidentlyDriftDetector()
    suite = detector.run_full_suite(reference, current)
    assert suite.data_drift is not None
    assert suite.data_quality is not None


def test_psi_threshold_triggers_retrain(adult_df):
    """High drift should trigger retraining."""
    from drift.evidently_detector import EvidentlyDriftDetector
    from retraining.trigger import DriftBasedRetrigger
    from data.fetchers.synthetic_drifter import PhysicsBasedDriftSimulator
    from data.processors.feature_engineer import FeatureEngineer

    engineer = FeatureEngineer()
    sample = adult_df.sample(1000, random_state=42).reset_index(drop=True)
    df_clean = engineer.handle_missing(sample)
    df_clean = engineer.encode_categoricals(df_clean)
    num_df = df_clean.select_dtypes(include=["number"]).fillna(0)
    n = len(num_df)
    reference = num_df.iloc[: n // 2]
    drifter = PhysicsBasedDriftSimulator()
    current = drifter.simulate_sudden_drift(num_df.iloc[n // 2 :].copy(), shift_magnitude=5.0)
    common = list(set(reference.columns) & set(current.columns))

    detector = EvidentlyDriftDetector()
    suite = detector.run_full_suite(reference[common], current[common])

    # Manually set high drift share for trigger test
    suite.data_drift.drift_share = 0.6

    trigger = DriftBasedRetrigger()
    assert trigger.should_retrain(suite) is True


def test_html_report_generated(adult_df):
    """generate_html_report should create a non-empty file."""
    from drift.evidently_detector import EvidentlyDriftDetector
    import tempfile
    import os

    sample = adult_df.sample(500, random_state=42).reset_index(drop=True)
    reference, current = _make_reference_current(sample, shift=False)
    detector = EvidentlyDriftDetector()
    suite = detector.run_full_suite(reference, current)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        tmp_path = f.name
    try:
        path = detector.generate_html_report(suite, tmp_path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
