from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from models.base_model import BaseModel
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)
set_seed(42)


class _RegMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralNetRegressor(BaseModel):
    def __init__(self):
        super().__init__("neural_net_regressor", "regression")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, params: Dict[str, Any]) -> None:
        self.hidden_dims = params.get("hidden_dims", [256, 128, 64])
        self.dropout = params.get("dropout", 0.3)
        self.lr = params.get("lr", 0.001)
        self.epochs = params.get("epochs", 50)
        self.batch_size = params.get("batch_size", 256)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]:
        X_t = torch.tensor(np.array(X_train, dtype=np.float32)).to(self.device)
        y_t = torch.tensor(np.array(y_train, dtype=np.float32)).to(self.device)
        input_dim = X_t.shape[1]
        self.model = _RegMLP(input_dim, self.hidden_dims, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.HuberLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                out = self.model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()
        self.is_trained = True
        preds = self.predict(X_train)
        metrics = {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, preds))),
            "train_r2": float(r2_score(y_train, preds)),
        }
        if X_val is not None:
            val_preds = self.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            metrics["val_r2"] = float(r2_score(y_val, val_preds))
        return metrics

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(np.array(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
        return out.cpu().numpy()

    def predict_proba(self, X) -> np.ndarray:
        return self.predict(X).reshape(-1, 1)

    def get_feature_importance(self) -> Dict[str, float]:
        return {}
