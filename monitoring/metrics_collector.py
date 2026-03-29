import os
import time
from typing import Dict, Any
import psutil
from utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """System and model metrics collection."""

    def get_system_metrics(self) -> Dict[str, float]:
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / 1e9,
            "disk_percent": psutil.disk_usage("/").percent,
            "disk_free_gb": psutil.disk_usage("/").free / 1e9,
        }

    def get_process_metrics(self) -> Dict[str, float]:
        proc = psutil.Process(os.getpid())
        return {
            "process_cpu_percent": proc.cpu_percent(interval=0.1),
            "process_memory_mb": proc.memory_info().rss / 1e6,
            "process_threads": proc.num_threads(),
        }

    def get_mlflow_metrics(self, tracking_uri: str = "http://localhost:5000") -> Dict[str, Any]:
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient(tracking_uri)
            experiments = client.search_experiments()
            active_runs = 0
            for exp in experiments:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string="status = 'RUNNING'",
                )
                active_runs += len(runs)
            return {"active_runs": active_runs, "experiments": len(experiments)}
        except Exception as e:
            logger.warning(f"MLflow metrics collection failed: {e}")
            return {"active_runs": 0, "experiments": 0}

    def collect_all(self) -> Dict[str, Any]:
        metrics = {}
        metrics.update(self.get_system_metrics())
        metrics.update(self.get_process_metrics())
        return metrics
