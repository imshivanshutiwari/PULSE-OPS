"""4 tests for model training."""


def _prepare_adult_data(adult_df, sample_n=2000):
    from data.processors.feature_engineer import FeatureEngineer
    from data.processors.data_splitter import DataSplitter

    engineer = FeatureEngineer()
    df = engineer.handle_missing(adult_df.sample(sample_n, random_state=42).reset_index(drop=True))
    df = engineer.encode_categoricals(df)
    splitter = DataSplitter()
    return splitter.split(df, "income")


def test_xgboost_trains_and_produces_predictions(adult_df):
    """XGBoost should train and produce predictions of correct shape."""
    from models.classification.gradient_boosting import XGBoostClassifier

    X_train, X_val, X_test, y_train, y_val, y_test = _prepare_adult_data(adult_df)
    model = XGBoostClassifier()
    model.build({"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1})
    metrics = model.fit(X_train, y_train, X_val, y_val)
    assert "train_accuracy" in metrics
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})


def test_model_accuracy_above_baseline(adult_df):
    """Trained XGBoost on adult should achieve > 0.75 accuracy."""
    from models.classification.gradient_boosting import XGBoostClassifier

    X_train, X_val, X_test, y_train, y_val, y_test = _prepare_adult_data(adult_df, 3000)
    model = XGBoostClassifier()
    model.build({"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1})
    metrics = model.fit(X_train, y_train, X_val, y_val)
    assert (
        metrics.get("val_accuracy", 0) > 0.75
    ), f"Expected >0.75, got {metrics.get('val_accuracy', 0):.4f}"


def test_lightgbm_trains_regression(wine_df):
    """LightGBM should train on wine quality regression."""
    from data.processors.feature_engineer import FeatureEngineer
    from data.processors.data_splitter import DataSplitter
    from models.regression.gradient_boosting import LightGBMRegressor

    engineer = FeatureEngineer()
    df = engineer.handle_missing(wine_df.copy())
    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(df, "quality")
    model = LightGBMRegressor()
    model.build({"n_estimators": 30, "num_leaves": 20, "learning_rate": 0.1})
    metrics = model.fit(X_train, y_train, X_val, y_val)
    assert "val_r2" in metrics
    assert metrics["val_r2"] > 0.0


def test_random_forest_classifier(adult_df):
    """Random Forest should train and predict correctly."""
    from models.classification.random_forest import RandomForestClassifier

    X_train, X_val, X_test, y_train, y_val, y_test = _prepare_adult_data(adult_df, 2000)
    model = RandomForestClassifier()
    model.build({"n_estimators": 20, "max_depth": 6})
    metrics = model.fit(X_train, y_train, X_val, y_val)
    assert "val_accuracy" in metrics
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
