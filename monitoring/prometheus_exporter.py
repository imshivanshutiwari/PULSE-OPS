from prometheus_client import Counter, Gauge, Histogram, start_http_server
from utils.logger import get_logger

logger = get_logger(__name__)

model_prediction_latency = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency in seconds",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
model_accuracy_gauge = Gauge(
    "model_accuracy_gauge",
    "Current model accuracy",
    ["model_name"],
)
model_drift_score = Gauge(
    "model_drift_score",
    "Current model drift score",
    ["model_name"],
)
data_quality_score = Gauge(
    "data_quality_score",
    "Data quality score",
    ["dataset_name"],
)
retraining_count = Counter(
    "retraining_count_total",
    "Total number of retraining events",
    ["model_name"],
)
feature_store_requests = Counter(
    "feature_store_requests_total",
    "Feature store request count",
    ["dataset_name"],
)
mlflow_active_runs = Gauge(
    "mlflow_active_runs",
    "Number of active MLflow runs",
)
prefect_flow_runs = Counter(
    "prefect_flow_runs_total",
    "Total Prefect flow runs",
    ["flow_name", "status"],
)


class PulseOpsPrometheusExporter:
    """Custom Prometheus metrics exporter for PULSE-OPS."""

    def __init__(self, port: int = 8001):
        self.port = port
        self._server_started = False

    def start_server(self, port: int = None) -> None:
        port = port or self.port
        if not self._server_started:
            try:
                start_http_server(port)
                self._server_started = True
                logger.info(f"Prometheus metrics server started on port {port}")
            except OSError as e:
                logger.warning(f"Prometheus server already running or port in use: {e}")

    def update_model_metrics(
        self, model_name: str, accuracy: float, drift_score_val: float
    ) -> None:
        model_accuracy_gauge.labels(model_name=model_name).set(accuracy)
        model_drift_score.labels(model_name=model_name).set(drift_score_val)

    def record_prediction(self, model_name: str, latency_seconds: float) -> None:
        model_prediction_latency.labels(model_name=model_name).observe(latency_seconds)

    def increment_retraining_counter(self, model_name: str) -> None:
        retraining_count.labels(model_name=model_name).inc()

    def update_data_quality(self, dataset_name: str, score: float) -> None:
        data_quality_score.labels(dataset_name=dataset_name).set(score)

    def increment_feature_store_requests(self, dataset_name: str) -> None:
        feature_store_requests.labels(dataset_name=dataset_name).inc()

    def update_mlflow_runs(self, count: int) -> None:
        mlflow_active_runs.set(count)

    def record_flow_run(self, flow_name: str, status: str) -> None:
        prefect_flow_runs.labels(flow_name=flow_name, status=status).inc()
