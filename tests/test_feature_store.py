"""4 tests for feature store."""


def test_offline_retrieval_returns_correct_shape(small_adult_df):
    """Offline store should return correct number of features."""
    from data.processors.feature_engineer import FeatureEngineer
    from pathlib import Path

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    parquet_path = raw_dir / "adult_features.parquet"
    if not parquet_path.exists():
        engineer = FeatureEngineer()
        df = engineer.handle_missing(small_adult_df.copy())
        df = engineer.encode_categoricals(df)
        df = engineer.add_event_timestamp(df)
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

    from feature_store.offline_store import OfflineStore

    store = OfflineStore()
    df = store.get_historical_features("adult", [])
    assert df.shape[1] > 5, f"Expected >5 cols, got {df.shape[1]}"
    assert len(df) > 100, f"Expected >100 rows, got {len(df)}"


def test_online_store_latency_under_10ms(small_adult_df):
    """100 online store lookups should average < 10ms."""
    from data.processors.feature_engineer import FeatureEngineer
    from pathlib import Path

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    parquet_path = raw_dir / "adult_features.parquet"
    if not parquet_path.exists():
        engineer = FeatureEngineer()
        df = engineer.handle_missing(small_adult_df.copy())
        df = engineer.encode_categoricals(df)
        df = engineer.add_event_timestamp(df)
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

    from feature_store.online_store import OnlineStore

    store = OnlineStore()
    for i in range(100):
        store.get_online_features("adult", i % 1000)
    mean_latency = store.get_mean_latency_ms()
    assert mean_latency < 10.0, f"Mean latency {mean_latency:.2f}ms exceeds 10ms"


def test_feature_freshness_tracked(small_adult_df):
    """event_timestamp column must be present in stored features."""
    from data.processors.feature_engineer import FeatureEngineer
    from pathlib import Path

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    parquet_path = raw_dir / "adult_features.parquet"
    if not parquet_path.exists():
        engineer = FeatureEngineer()
        df = engineer.handle_missing(small_adult_df.copy())
        df = engineer.encode_categoricals(df)
        df = engineer.add_event_timestamp(df)
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

    from feature_store.offline_store import OfflineStore

    store = OfflineStore()
    df = store.get_historical_features("adult", [])
    assert "event_timestamp" in df.columns, "event_timestamp column missing"


def test_feature_pipeline_get_training_dataset(small_adult_df):
    """FeaturePipeline should return X, y for adult dataset."""
    from data.processors.feature_engineer import FeatureEngineer
    from pathlib import Path

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    parquet_path = raw_dir / "adult_features.parquet"
    if not parquet_path.exists():
        engineer = FeatureEngineer()
        df = engineer.handle_missing(small_adult_df.copy())
        df = engineer.encode_categoricals(df)
        df = engineer.add_event_timestamp(df)
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

    from feature_store.feature_pipeline import FeaturePipeline

    fp = FeaturePipeline()
    X, y = fp.get_training_dataset("adult", "income")
    assert X is not None
    assert y is not None
    assert len(X) > 100
