"""Central Prefect flows registry — imports and re-exports all flows."""

from orchestration.training_flow import (
    training_flow,
    load_features_task,
    split_data_task,
    hpo_task,
    train_model_task,
    register_model_task,
)
from orchestration.drift_flow import (
    drift_detection_flow,
    load_reference_data_task,
    load_current_data_task,
    run_evidently_task,
    store_drift_report_task,
    should_retrain_task,
)
from orchestration.retraining_flow import (
    retraining_flow,
    fetch_fresh_data_task,
    validate_data_task,
    run_retraining_task,
    evaluation_gate_task,
    promote_to_production_task,
    rollback_task,
    notify_task,
)
from orchestration.deployment_flow import (
    deployment_flow,
    load_production_model_task,
    health_check_task,
    deploy_to_serving_task,
)

__all__ = [
    "training_flow",
    "drift_detection_flow",
    "retraining_flow",
    "deployment_flow",
]
