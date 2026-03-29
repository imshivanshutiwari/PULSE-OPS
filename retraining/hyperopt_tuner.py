from typing import Any, Dict
import numpy as np
import optuna
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from utils.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaHPOTuner:
    """Optuna hyperparameter optimization for all model types."""

    def __init__(self, n_trials: int = 50, cv_folds: int = 3):
        self.n_trials = n_trials
        self.cv_folds = cv_folds

    def tune_xgboost(self, X_train, y_train, task: str = "classification") -> Dict[str, Any]:
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "use_label_encoder": False,
                "eval_metric": "logloss" if task == "classification" else "rmse",
                "random_state": 42,
                "n_jobs": -1,
            }
            if task == "classification":
                model = xgb.XGBClassifier(**params)
                scores = cross_val_score(
                    model, X_train, y_train, cv=self.cv_folds, scoring="accuracy"
                )
                return scores.mean()
            else:
                model = xgb.XGBRegressor(
                    **{
                        k: v
                        for k, v in params.items()
                        if k not in ["use_label_encoder", "eval_metric"]
                    }
                )
                scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring="r2")
                return scores.mean()

        direction = "maximize"
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.log_to_mlflow(study, f"xgboost_{task}_hpo")
        return study.best_params

    def tune_lightgbm(self, X_train, y_train, task: str = "regression") -> Dict[str, Any]:
        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            if task == "classification":
                model = lgb.LGBMClassifier(**params)
                scores = cross_val_score(
                    model, X_train, y_train, cv=self.cv_folds, scoring="accuracy"
                )
            else:
                model = lgb.LGBMRegressor(**params)
                scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring="r2")
            return scores.mean()

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.log_to_mlflow(study, f"lightgbm_{task}_hpo")
        return study.best_params

    def tune_neural_net(self, X_train, y_train, task: str = "classification") -> Dict[str, Any]:
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        def objective(trial):
            n_layers = trial.suggest_int("n_layers", 2, 4)
            hidden_dims = tuple(
                trial.suggest_int(f"n_units_l{i}", 32, 512) for i in range(n_layers)
            )
            params = {
                "hidden_layer_sizes": hidden_dims,
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "max_iter": 50,
                "random_state": 42,
            }
            if task == "classification":
                model = MLPClassifier(**params)
                scores = cross_val_score(
                    model, X_train, y_train, cv=self.cv_folds, scoring="accuracy"
                )
            else:
                model = MLPRegressor(**params)
                scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring="r2")
            return scores.mean()

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=min(self.n_trials, 30), show_progress_bar=False)
        self.log_to_mlflow(study, f"neural_net_{task}_hpo")
        return study.best_params

    def log_to_mlflow(self, study: optuna.Study, model_name: str) -> None:
        try:
            with mlflow.start_run(run_name=f"hpo_{model_name}", nested=True):
                mlflow.log_params(study.best_params)
                mlflow.log_metric("best_value", study.best_value)
                mlflow.log_metric("n_trials", len(study.trials))
                logger.info(f"HPO logged to MLflow: {model_name}, best={study.best_value:.4f}")
        except Exception as e:
            logger.warning(f"MLflow HPO logging failed: {e}")
