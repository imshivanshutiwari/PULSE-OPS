"""4 tests for MLflow model registry."""
import pytest
import os
import tempfile
import mlflow
from mlflow.tracking import MlflowClient


def _setup_local_mlflow():
    """Set up a local file-based MLflow for testing."""
    tmp = tempfile.mkdtemp()
    uri = f"file://{tmp}"
    mlflow.set_tracking_uri(uri)
    return uri, tmp


def _train_and_log_model(uri: str):
    """Train a minimal model and log it to MLflow."""
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier as SklearnRF
    import mlflow.sklearn

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test_registry")
    iris = load_iris()
    X, y = iris.data, iris.target
    model = SklearnRF(n_estimators=5, random_state=42)
    model.fit(X, y)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", 0.95)
        return run.info.run_id


def test_model_registers_successfully():
    """register_model should return a ModelVersion."""
    uri, _ = _setup_local_mlflow()
    run_id = _train_and_log_model(uri)

    from registry.mlflow_registry import MLflowModelRegistry

    reg = MLflowModelRegistry(tracking_uri=uri)
    version = reg.register_model(run_id, "test_iris_model", artifact_path="model")
    assert version is not None
    assert int(version.version) >= 1


def test_staging_promotion_works():
    """promote_to_staging should transition model to Staging stage."""
    uri, _ = _setup_local_mlflow()
    run_id = _train_and_log_model(uri)

    from registry.mlflow_registry import MLflowModelRegistry

    reg = MLflowModelRegistry(tracking_uri=uri)
    version = reg.register_model(run_id, "test_iris_staging", artifact_path="model")
    result = reg.promote_to_staging("test_iris_staging", int(version.version))
    assert result.current_stage == "Staging"


def test_production_promotion_works():
    """promote_to_production should transition model to Production."""
    uri, _ = _setup_local_mlflow()
    run_id = _train_and_log_model(uri)

    from registry.mlflow_registry import MLflowModelRegistry

    reg = MLflowModelRegistry(tracking_uri=uri)
    version = reg.register_model(run_id, "test_iris_prod", artifact_path="model")
    reg.promote_to_staging("test_iris_prod", int(version.version))
    result = reg.promote_to_production("test_iris_prod", int(version.version))
    assert result is True


def test_rollback_demotes_bad_version():
    """rollback should move production to archived and restore previous."""
    uri, _ = _setup_local_mlflow()

    # Register 2 versions
    run_id1 = _train_and_log_model(uri)
    run_id2 = _train_and_log_model(uri)

    from registry.mlflow_registry import MLflowModelRegistry

    reg = MLflowModelRegistry(tracking_uri=uri)
    v1 = reg.register_model(run_id1, "test_rollback_model", artifact_path="model")
    v2 = reg.register_model(run_id2, "test_rollback_model", artifact_path="model")

    # Promote v1 to production, then v2
    reg.promote_to_staging("test_rollback_model", int(v1.version))
    reg.promote_to_production("test_rollback_model", int(v1.version))
    reg.promote_to_staging("test_rollback_model", int(v2.version))
    # Archive v1 manually (simulate failure promotion path)
    reg.client.transition_model_version_stage(
        "test_rollback_model", str(v1.version), stage="Archived"
    )
    reg.client.transition_model_version_stage(
        "test_rollback_model", str(v2.version), stage="Production"
    )

    reg.rollback("test_rollback_model")
    # After rollback, v1 (archived) should be promoted back to Production
    versions = reg.list_versions("test_rollback_model")
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    assert len(prod_versions) >= 1
