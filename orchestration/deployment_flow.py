from prefect import flow, task
from utils.logger import get_logger

logger = get_logger(__name__)


@task(name="load-production-model", retries=2)
def load_production_model_task(model_name: str):
    from registry.mlflow_registry import MLflowModelRegistry

    try:
        reg = MLflowModelRegistry()
        model = reg.get_production_model(model_name)
        return model
    except Exception as e:
        logger.warning(f"Could not load production model: {e}")
        return None


@task(name="health-check")
def health_check_task(model, model_name: str):
    if model is None:
        logger.warning(f"No production model found for {model_name}")
        return False
    try:
        import numpy as np

        test_input = np.zeros((1, 10))
        model.predict(test_input)
        logger.info(f"Health check passed for {model_name}")
        return True
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        return False


@task(name="deploy-to-serving")
def deploy_to_serving_task(model, model_name: str):
    logger.info(f"Model {model_name} is available for serving via FastAPI")
    return {"status": "deployed", "model_name": model_name}


@flow(name="deployment-flow", description="Model deployment to serving infrastructure")
def deployment_flow(model_name: str = "xgboost_classifier", model_version: int = None):
    logger.info(f"Deployment flow: {model_name}")
    model = load_production_model_task(model_name)
    healthy = health_check_task(model, model_name)
    if not healthy:
        logger.error(f"Model {model_name} failed health check, aborting deployment")
        return {"status": "failed", "reason": "health_check_failed"}
    result = deploy_to_serving_task(model, model_name)
    return result
