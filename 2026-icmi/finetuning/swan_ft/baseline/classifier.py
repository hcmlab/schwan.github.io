"""Classifiers for the feature-extraction baseline."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class LogRegClassifier:
    """Scikit-learn logistic regression wrapper."""

    def __init__(self, C: float = 1.0, class_weight: str = "balanced"):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight=class_weight,
            solver="lbfgs",
        )

    def fit(self, X: np.ndarray, y: list[str]) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> list[str]:
        return self.model.predict(X).tolist()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> None:
        with path.open("rb") as f:
            self.model = pickle.load(f)


class MLPClassifier:
    """Simple 2-layer MLP trained with PyTorch."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        epochs: int = 50,
        lr: float = 1e-3,
    ):
        import torch
        import torch.nn as nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.lr = lr

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        ).to(self.device)

        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}

    def fit(self, X: np.ndarray, y: list[str], X_val: np.ndarray | None = None, y_val: list[str] | None = None) -> None:
        import torch
        import torch.nn as nn

        # Build label mapping
        unique_labels = sorted(set(y))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        # Update output layer if num_classes changed
        actual_classes = len(unique_labels)
        if self.net[-1].out_features != actual_classes:
            in_features = self.net[-1].in_features
            self.net[-1] = nn.Linear(in_features, actual_classes).to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor([self.label_to_idx[label] for label in y], dtype=torch.long, device=self.device)

        # Compute class weights for balanced loss
        class_counts = np.bincount([self.label_to_idx[label] for label in y], minlength=actual_classes)
        class_weights = 1.0 / np.maximum(class_counts.astype(float), 1.0)
        class_weights = class_weights / class_weights.sum() * actual_classes
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_t = torch.tensor([self.label_to_idx.get(l, 0) for l in y_val], dtype=torch.long, device=self.device)

        self.net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.net(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

            if has_val and (epoch + 1) % 5 == 0:
                self.net.eval()
                with torch.no_grad():
                    val_logits = self.net(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                self.net.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

    def predict(self, X: np.ndarray) -> list[str]:
        import torch

        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.net(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return [self.idx_to_label.get(int(p), "(UNKNOWN)") for p in preds]

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
        }, path)

    def load(self, path: Path) -> None:
        import torch
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.label_to_idx = checkpoint["label_to_idx"]
        self.idx_to_label = checkpoint["idx_to_label"]
        self.net.load_state_dict(checkpoint["state_dict"])


def pool_features(frame_features: np.ndarray, method: str = "mean") -> np.ndarray:
    """Aggregate per-frame features into a single vector.

    Args:
        frame_features: (N_frames, D) array
        method: "mean" or "mean_std"

    Returns:
        (D,) for mean, (2*D,) for mean_std
    """
    if method == "mean":
        return frame_features.mean(axis=0)
    if method == "mean_std":
        mu = frame_features.mean(axis=0)
        std = frame_features.std(axis=0)
        return np.concatenate([mu, std])
    raise ValueError(f"Unknown pooling method: {method}")
