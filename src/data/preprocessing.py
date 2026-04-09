"""Data preprocessing: scaling, feature extraction."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger


class WindowMinMaxScaler:
    """Rolling window min-max scaler. Pricing-agnostic."""

    def __init__(self, window_size: int = 120, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon

    def transform(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n, f = data.shape
        scaled = np.zeros_like(data)
        for i in range(n):
            start = max(0, i - self.window_size + 1)
            window = data[start : i + 1]
            w_min = window.min(axis=0)
            w_max = window.max(axis=0)
            scaled[i] = (data[i] - w_min) / (w_max - w_min + self.epsilon)
        return scaled


class ZScoreScaler:
    """Rolling z-score normalization."""

    def __init__(self, window_size: int = 120, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon

    def transform(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n, f = data.shape
        scaled = np.zeros_like(data)
        for i in range(n):
            start = max(0, i - self.window_size + 1)
            window = data[start : i + 1]
            scaled[i] = (data[i] - window.mean(axis=0)) / (window.std(axis=0) + self.epsilon)
        return scaled


def get_scaler(method: str, window_size: int):
    if method == "window_minmax":
        return WindowMinMaxScaler(window_size)
    elif method == "zscore":
        return ZScoreScaler(window_size)
    raise ValueError(f"Unknown scaler: {method}")


def prepare_features(
    df: pl.DataFrame,
    scaler_method: str = "window_minmax",
    window_size: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and scale OHLCV+spread features.

    Returns:
        features: (n_samples, 6) scaled features
        close_prices: (n_samples,) raw close prices (for labeling)
    """
    # Clean weekends
    df = df.filter(pl.col("timestamp").dt.weekday().is_in([1, 2, 3, 4, 5]))
    # Forward fill gaps
    for c in ["open", "high", "low", "close"]:
        df = df.with_columns(pl.col(c).forward_fill())

    feature_cols = ["open", "high", "low", "close", "tick_volume", "spread"]
    features = df.select(feature_cols).to_numpy().astype(np.float32)
    close_prices = df["close"].to_numpy()

    scaler = get_scaler(scaler_method, window_size)
    features_scaled = scaler.transform(features).astype(np.float32)

    logger.info(f"Features prepared: {features_scaled.shape}")
    return features_scaled, close_prices
