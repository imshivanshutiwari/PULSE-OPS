"""4 tests for retraining."""

import pytest
import pandas as pd
import numpy as np


def test_trigger_fires_on_high_drift(drift_suite):
    """Trigger should fire when drift_share > threshold."""
    from retraining.trigger import DriftBasedRetrigger

    drift_suite.data_drift.drift_share = 0.8
    trigger = DriftBasedRetrigger()
    assert trigger.should_retrain(drift_suite) is True


def test_evaluation_gate_passes_better_model():
    """Gate should pass when new model is better than current."""
    from retraining.evaluation_gate import EvaluationGate

    gate = EvaluationGate()
    new_metrics = {"val_accuracy": 0.87}
    current_metrics = {"val_accuracy": 0.82}
    assert gate.check(new_metrics, current_metrics) is True


def test_evaluation_gate_fails_worse_model():
    """Gate should fail when new model is significantly worse."""
    from retraining.evaluation_gate import EvaluationGate

    gate = EvaluationGate()
    new_metrics = {"val_accuracy": 0.70}
    current_metrics = {"val_accuracy": 0.85}
    assert gate.check(new_metrics, current_metrics) is False


def test_urgency_computation(drift_suite):
    """Urgency should reflect drift level correctly."""
    from retraining.trigger import DriftBasedRetrigger

    trigger = DriftBasedRetrigger()

    drift_suite.data_drift.drift_share = 0.6
    assert trigger.compute_urgency(drift_suite) == "CRITICAL"

    drift_suite.data_drift.drift_share = 0.35
    assert trigger.compute_urgency(drift_suite) == "HIGH"

    drift_suite.data_drift.drift_share = 0.1
    urgency = trigger.compute_urgency(drift_suite)
    assert urgency in {"MEDIUM", "LOW"}
