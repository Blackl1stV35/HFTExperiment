"""Confidence calibration via isotonic regression — Phase 3 v2.

Implements §5.1 of IMPROVEMENTS.md (Variance≠Importance):
    - Confidence should regress realised Sharpe over next N bars, not entropy
    - Activation variance is 96% uncorrelated with output importance
    - Calibrate via isotonic regression on a validation split

Usage:
    from src.training.confidence_calibration import (
        SharpeConfidenceHead, IsotonicCalibrator
    )

    # During training: add SharpeConfidenceHead as auxiliary loss
    # Post-training: fit IsotonicCalibrator on val split
    # At inference: apply calibrator before passing to RL agent

The RL agent's confidence gate (0.70) and reward scaling should use the
calibrated confidence, not the raw MSE-trained head output.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class SharpeConfidenceHead(nn.Module):
    """Predicts realised Sharpe over next N bars from intermediate encoder embedding.

    §5.2 (Variance≠Importance): tap confidence from the 60-70% depth point
    of the encoder (feature construction phase), not the final layer
    (linear refinement phase, near-deterministic).

    The head is trained with an auxiliary loss alongside the primary CE loss:
        L_total = L_CE + alpha * L_sharpe_conf
    where L_sharpe_conf = MSE(pred_sharpe, realised_sharpe_next_N)

    Args:
        in_dim:    embedding dimension from intermediate encoder layer
        n_bars:    horizon for Sharpe calculation (default 20 M1 bars = 20 min)
        alpha:     loss weight for confidence head (default 0.10)
    """

    def __init__(self, in_dim: int, n_bars: int = 20, alpha: float = 0.10):
        super().__init__()
        self.n_bars = n_bars
        self.alpha  = alpha
        self.head   = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, intermediate_embedding: torch.Tensor) -> torch.Tensor:
        """Returns predicted confidence (Sharpe proxy) in [0, 1]."""
        return self.head(intermediate_embedding).squeeze(-1)

    @staticmethod
    def compute_realised_sharpe(
        close_prices: np.ndarray,
        idx:          np.ndarray,
        n_bars:       int = 20,
    ) -> np.ndarray:
        """Compute realised Sharpe over next n_bars for each sequence index.

        Used to generate confidence targets during training.
        Returns values in [0, 1] via sigmoid normalisation.
        """
        targets = np.zeros(len(idx), dtype=np.float32)
        for i, seq_end in enumerate(idx):
            future_end = min(seq_end + n_bars, len(close_prices) - 1)
            if future_end <= seq_end:
                targets[i] = 0.5
                continue
            prices  = close_prices[seq_end : future_end + 1]
            returns = np.diff(prices) / (prices[:-1] + 1e-8)
            if len(returns) == 0 or returns.std() < 1e-8:
                targets[i] = 0.5
                continue
            sharpe   = returns.mean() / returns.std() * np.sqrt(252 * 390)
            # Normalise: sigmoid maps Sharpe=0 → 0.5, positive → >0.5
            targets[i] = float(1.0 / (1.0 + np.exp(-sharpe / 2.0)))
        return targets


class IsotonicCalibrator:
    """Post-hoc isotonic regression calibrator for confidence outputs.

    §5.1: calibrate raw confidence against realised profitability on a
    held-out validation split. Isotonic regression is monotone and does
    not assume a parametric form.

    Usage:
        cal = IsotonicCalibrator()
        cal.fit(raw_confidences_val, realised_sharpes_val)
        calibrated = cal.transform(raw_confidences_test)
    """

    def __init__(self):
        self._calibrator = None

    def fit(self, raw_conf: np.ndarray, realised: np.ndarray) -> None:
        """Fit isotonic regression from raw confidence to realised Sharpe proxy."""
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            raise ImportError("sklearn required for calibration: pip install scikit-learn")
        order = np.argsort(raw_conf)
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_conf[order], realised[order])

    def transform(self, raw_conf: np.ndarray) -> np.ndarray:
        if self._calibrator is None:
            raise RuntimeError("Call fit() before transform()")
        return self._calibrator.predict(raw_conf).astype(np.float32)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._calibrator, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            self._calibrator = pickle.load(f)
