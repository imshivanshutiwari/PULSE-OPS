import random
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
