from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_config(name: str) -> dict:
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_pipeline_config() -> dict:
    return load_config("pipeline_config")


def get_drift_config() -> dict:
    return load_config("drift_config")


def get_model_config() -> dict:
    return load_config("model_config")


def get_monitoring_config() -> dict:
    return load_config("monitoring_config")
