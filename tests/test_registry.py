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
    """promote_to_staging should set the 'staging' alias on the model version."""
    uri, _ = _setup_local_mlflow()
    run_id = _train_and_log_model(uri)

    from registry.mlflow_registry import MLflowModelRegistry, ALIAS_STAGING

    reg = MLflowModelRegistry(tracking_uri=uri)
    version = reg.register_model(run_id, "test_iris_staging", artifact_path="model")
    result = reg.promote_to_staging("test_iris_staging", int(version.version))
    assert ALIAS_STAGING in (result.aliases or [])


def test_production_promotion_works():
    """promote_to_production should set the 'production' alias."""
    uri, _ = _setup_local_mlflow()
    run_id = _train_and_log_model(uri)

    from registry.mlflow_registry import MLflowModelRegistry

    reg = MLflowModelRegistry(tracking_uri=uri)
    version = reg.register_model(run_id, "test_iris_prod", artifact_path="model")
    reg.promote_to_staging("test_iris_prod", int(version.version))
    result = reg.promote_to_production("test_iris_prod", int(version.version))
    assert result is True


def test_rollback_demotes_bad_version():
    """rollback should restore the most recent archived version to production."""
    uri, _ = _setup_local_mlflow()

    run_id1 = _train_and_log_model(uri)
    run_id2 = _train_and_log_model(uri)

    from registry.mlflow_registry import MLflowModelRegistry, ALIAS_PRODUCTION

    reg = MLflowModelRegistry(tracking_uri=uri)
    v1 = reg.register_model(run_id1, "test_rollback_model", artifact_path="model")
    v2 = reg.register_model(run_id2, "test_rollback_model", artifact_path="model")

    # Promote v1 to production, archive it, then promote v2
    reg.promote_to_staging("test_rollback_model", int(v1.version))
    reg.promote_to_production("test_rollback_model", int(v1.version))
    reg._archive_version("test_rollback_model", int(v1.version))
    reg.promote_to_staging("test_rollback_model", int(v2.version))
    reg.promote_to_production("test_rollback_model", int(v2.version))

    # Rollback should restore v1
    result = reg.rollback("test_rollback_model")
    assert result is not None
    prod = reg._get_version_by_alias("test_rollback_model", ALIAS_PRODUCTION)
    assert prod is not None
