import time
from typing import Dict, Any
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

SERVICES = {
    "mlflow": "http://localhost:5000/health",
    "prefect": "http://localhost:4200/api/health",
    "prometheus": "http://localhost:9090/-/healthy",
    "grafana": "http://localhost:3000/api/health",
    "redis": None,
    "model_serving": "http://localhost:8000/health",
}


class HealthChecker:
    """Service health checks for PULSE-OPS infrastructure."""

    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout

    def check_service(self, name: str, url: str = None) -> Dict[str, Any]:
        if url is None:
            if name == "redis":
                return self._check_redis()
            return {"service": name, "status": "unknown", "latency_ms": 0.0}
        t0 = time.time()
        try:
            resp = requests.get(url, timeout=self.timeout)
            latency = (time.time() - t0) * 1000
            return {
                "service": name,
                "status": "healthy" if resp.status_code < 400 else "degraded",
                "latency_ms": latency,
                "status_code": resp.status_code,
            }
        except requests.exceptions.RequestException as e:
            return {
                "service": name,
                "status": "offline",
                "latency_ms": (time.time() - t0) * 1000,
                "error": str(e),
            }

    def _check_redis(self) -> Dict[str, Any]:
        t0 = time.time()
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=2)
            r.ping()
            return {
                "service": "redis",
                "status": "healthy",
                "latency_ms": (time.time() - t0) * 1000,
            }
        except Exception as e:
            return {
                "service": "redis",
                "status": "offline",
                "error": str(e),
                "latency_ms": (time.time() - t0) * 1000,
            }

    def check_all(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        for name, url in SERVICES.items():
            results[name] = self.check_service(name, url)
        return results

    def compute_health_score(self, checks: Dict[str, Dict]) -> str:
        statuses = [v["status"] for v in checks.values()]
        healthy = statuses.count("healthy")
        total = len(statuses)
        ratio = healthy / total
        if ratio >= 0.9:
            return "A"
        elif ratio >= 0.75:
            return "B"
        elif ratio >= 0.6:
            return "C"
        elif ratio >= 0.4:
            return "D"
        return "F"
