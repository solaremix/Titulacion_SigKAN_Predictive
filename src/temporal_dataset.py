"""
Temporal dataset utilities for IAQ contaminant prediction.
Provides IAQTemporalPreprocessor (70/15/15 split, shuffle=False)
and SlidingWindowDataset for sequence modelling.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple


# Target contaminants
IAQ_TARGETS = ["CO2", "PM2.5", "PM10", "TVOC"]


class IAQTemporalPreprocessor:
    """
    Preprocesses IAQ time-series data with a strict temporal split.

    Split: 70% train / 15% val / 15% test  (no shuffle — preserves time order).
    Scales features with StandardScaler fitted only on train set.

    Args:
        target_cols:  list of target column names (default: IAQ_TARGETS)
        feature_cols: explicit feature columns; None → all non-target columns
    """

    def __init__(
        self,
        target_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.target_cols = target_cols or IAQ_TARGETS
        self.feature_cols = feature_cols
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],   # (X_train, y_train)
        Tuple[np.ndarray, np.ndarray],   # (X_val,   y_val)
        Tuple[np.ndarray, np.ndarray],   # (X_test,  y_test)
    ]:
        """Fit on train split, transform all splits. Returns (train, val, test)."""
        feature_cols = self.feature_cols or [
            c for c in df.columns if c not in self.target_cols
        ]
        self.feature_cols_ = feature_cols

        X = df[feature_cols].values.astype(np.float32)
        y = df[self.target_cols].values.astype(np.float32)

        n = len(X)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)

        X_train, y_train = X[:n_train],          y[:n_train]
        X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test,  y_test  = X[n_train + n_val:],  y[n_train + n_val:]

        # Fit scalers only on train
        X_train = self.scaler_X.fit_transform(X_train)
        X_val   = self.scaler_X.transform(X_val)
        X_test  = self.scaler_X.transform(X_test)

        y_train = self.scaler_y.fit_transform(y_train)
        y_val   = self.scaler_y.transform(y_val)
        y_test  = self.scaler_y.transform(y_test)

        self._fitted = True
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Undo target scaling for interpretable predictions."""
        assert self._fitted, "Call fit_transform first"
        return self.scaler_y.inverse_transform(y_scaled)


class SlidingWindowDataset(Dataset):
    """
    PyTorch Dataset that produces overlapping sliding windows.

    Args:
        X:          feature array (N, F)
        y:          target array  (N, T)
        window:     look-back steps (input sequence length)
        horizon:    steps ahead to predict (default 1)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window: int = 24,
        horizon: int = 1,
    ):
        assert len(X) == len(y), "X and y must have same length"
        assert window + horizon <= len(X), "Not enough data for window + horizon"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window = window
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.X) - self.window - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.window]            # (window, F)
        y_tgt = self.y[idx + self.window + self.horizon - 1]  # (T,)
        return x_seq, y_tgt
