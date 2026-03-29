from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from models.base_model import BaseModel
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)
set_seed(42)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralNetClassifier(BaseModel):
    def __init__(self):
        super().__init__("neural_net_classifier", "classification")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, params: Dict[str, Any]) -> None:
        self.hidden_dims = params.get("hidden_dims", [256, 128, 64])
        self.dropout = params.get("dropout", 0.3)
        self.lr = params.get("lr", 0.001)
        self.epochs = params.get("epochs", 50)
        self.batch_size = params.get("batch_size", 256)
        self._params = params

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]:
        X_t = torch.tensor(np.array(X_train, dtype=np.float32)).to(self.device)
        y_t = torch.tensor(np.array(y_train, dtype=np.int64)).to(self.device)
        n_classes = int(y_t.max().item()) + 1
        input_dim = X_t.shape[1]
        self.model = _MLP(input_dim, self.hidden_dims, n_classes, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                out = self.model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
        self.is_trained = True
        train_preds = self.predict(X_train)
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_preds)),
            "train_f1": float(f1_score(y_train, train_preds, average="weighted")),
        }
        if X_val is not None:
            val_preds = self.predict(X_val)
            metrics["val_accuracy"] = float(accuracy_score(y_val, val_preds))
            metrics["val_f1"] = float(f1_score(y_val, val_preds, average="weighted"))
        return metrics

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(np.array(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(np.array(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            proba = torch.softmax(logits, dim=1)
        return proba.cpu().numpy()

    def get_feature_importance(self) -> Dict[str, float]:
        return {}
