from datetime import datetime
from typing import Optional
import mlflow
from prefect import flow, task
from utils.logger import get_logger

logger = get_logger(__name__)


@task(name="load-features", retries=2, retry_delay_seconds=10)
def load_features_task(dataset_name: str):
    from feature_store.feature_pipeline import FeaturePipeline

    target_map = {
        "adult": "income",
        "wine_quality": "quality",
        "bike_sharing": "cnt",
        "german_credit": "creditworthiness",
    }
    fp = FeaturePipeline()
    target = target_map.get(dataset_name, "target")
    X, y = fp.get_training_dataset(dataset_name, target)
    return X, y, target


@task(name="split-data")
def split_data_task(X, y, target_col: str):
    import pandas as pd
    from data.processors.data_splitter import DataSplitter

    df = pd.concat([X, y], axis=1)
    splitter = DataSplitter()
    return splitter.split(df, target_col)


@task(name="hpo")
def hpo_task(X_train, y_train, model_type: str):
    from retraining.hyperopt_tuner import OptunaHPOTuner

    tuner = OptunaHPOTuner(n_trials=10)
    task_type = "classification" if "classifier" in model_type else "regression"
    if "xgboost" in model_type:
        return tuner.tune_xgboost(X_train.fillna(0), y_train, task=task_type)
    elif "lightgbm" in model_type:
        return tuner.tune_lightgbm(X_train.fillna(0), y_train, task=task_type)
    else:
        return tuner.tune_neural_net(X_train.fillna(0), y_train, task=task_type)


@task(name="train-model")
def train_model_task(model_type: str, X_train, y_train, X_val, y_val, params: dict):
    from models.trainer import ModelTrainer
    from models.classification.gradient_boosting import XGBoostClassifier, LightGBMClassifier
    from models.regression.gradient_boosting import XGBoostRegressor, LightGBMRegressor

    model_map = {
        "xgboost_classifier": XGBoostClassifier,
        "xgboost_regressor": XGBoostRegressor,
        "lightgbm_regressor": LightGBMRegressor,
        "lightgbm_classifier": LightGBMClassifier,
    }
    ModelClass = model_map.get(model_type, XGBoostClassifier)
    model = ModelClass()
    trainer = ModelTrainer()
    result = trainer.train(model, X_train.fillna(0), y_train, X_val.fillna(0), y_val, params=params)
    return result


@task(name="register-model")
def register_model_task(run_id: str, model_type: str):
    from registry.mlflow_registry import MLflowModelRegistry

    reg = MLflowModelRegistry()
    try:
        version = reg.register_model(run_id, model_type)
        reg.promote_to_staging(model_type, int(version.version))
        return int(version.version)
    except Exception as e:
        logger.warning(f"Registration failed: {e}")
        return None


@flow(name="training-flow", description="Train and register a new model version")
def training_flow(
    dataset_name: str = "adult", model_type: str = "xgboost_classifier", run_hpo: bool = True
):
    logger.info(f"Training flow: dataset={dataset_name}, model={model_type}")
    X, y, target_col = load_features_task(dataset_name)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_task(X, y, target_col)
    if run_hpo:
        best_params = hpo_task(X_train, y_train, model_type)
    else:
        from utils.config_loader import get_model_config

        cfg = get_model_config()
        key = model_type.replace("_classifier", "").replace("_regressor", "")
        best_params = cfg.get(key, {})
    result = train_model_task(model_type, X_train, y_train, X_val, y_val, best_params)
    version = register_model_task(result["run_id"], model_type)
    logger.info(f"Training complete: run_id={result['run_id']}, version={version}")
    return {"run_id": result["run_id"], "metrics": result["metrics"], "version": version}


if __name__ == "__main__":
    from utils.config_loader import get_pipeline_config

    cfg = get_pipeline_config()
    for dataset, model in cfg["active_models"].items():
        training_flow(dataset_name=dataset, model_type=model, run_hpo=False)
