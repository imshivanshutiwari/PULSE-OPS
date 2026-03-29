from prefect import flow, task
from utils.logger import get_logger

logger = get_logger(__name__)


@task(name="fetch-fresh-data", retries=2)
def fetch_fresh_data_task(dataset_name: str):
    from feature_store.feature_pipeline import FeaturePipeline

    fp = FeaturePipeline()
    return fp.compute_features(dataset_name)


@task(name="validate-data")
def validate_data_task(data):
    from data.processors.data_validator import DataValidator

    validator = DataValidator()
    result = validator.validate(data)
    if not result["overall_passed"]:
        logger.warning(f"Data validation issues: {result}")
    return result["overall_passed"]


@task(name="run-retraining")
def run_retraining_task(dataset_name: str, model_type: str, drift_suite=None):
    from retraining.retraining_pipeline import RetrainingPipeline

    pipeline = RetrainingPipeline()
    return pipeline.run(dataset_name, model_type, drift_suite=drift_suite, force=True)


@task(name="evaluate-gate")
def evaluation_gate_task(result: dict):
    return result.get("gate_passed", True)


@task(name="promote-to-production")
def promote_to_production_task(model_type: str, version: int):
    from registry.mlflow_registry import MLflowModelRegistry

    reg = MLflowModelRegistry()
    try:
        versions = reg.list_versions(model_type)
        staging = [v for v in versions if v.current_stage == "Staging"]
        prod = [v for v in versions if v.current_stage == "Production"]
        champion_version = int(prod[0].version) if prod else None
        if staging:
            reg.promote_to_production(model_type, int(staging[0].version), champion_version)
            return True
    except Exception as e:
        logger.warning(f"Promotion failed: {e}")
    return False


@task(name="rollback")
def rollback_task(model_type: str):
    from registry.rollback_manager import RollbackManager

    rm = RollbackManager()
    return rm.rollback_to_previous(model_type)


@task(name="notify")
def notify_task(message: str):
    logger.info(f"NOTIFICATION: {message}")
    return message


@flow(name="retraining-flow", description="Self-healing retraining pipeline")
def retraining_flow(
    dataset_name: str = "adult",
    model_type: str = "xgboost_classifier",
    triggered_by: str = "manual",
    drift_suite=None,
):
    notify_task(f"Retraining triggered by: {triggered_by} for {dataset_name}/{model_type}")
    data = fetch_fresh_data_task(dataset_name)
    valid = validate_data_task(data)
    result = run_retraining_task(dataset_name, model_type, drift_suite)
    gate_passed = evaluation_gate_task(result)
    if gate_passed and result.get("version") is not None:
        promoted = promote_to_production_task(model_type, result.get("version", 1))
        if promoted:
            notify_task(f"New model promoted to Production: {model_type}")
        else:
            notify_task(f"Promotion skipped: {model_type}")
    else:
        rollback_task(model_type)
        notify_task(f"Gate failed, keeping current model: {model_type}")
    return {
        "status": result.get("status"),
        "gate_passed": gate_passed,
        "run_id": result.get("run_id"),
        "metrics": result.get("metrics"),
    }


if __name__ == "__main__":
    retraining_flow()
