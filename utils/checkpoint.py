import json
import pickle
from pathlib import Path
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(obj, path: str, metadata: dict = None) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    if metadata:
        meta_path = p.with_suffix(".json")
        metadata["saved_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    logger.info(f"Checkpoint saved: {path}")
    return str(p)


def load_checkpoint(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(p, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Checkpoint loaded: {path}")
    return obj
