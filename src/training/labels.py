"""3-class rule-based labeling: triple barrier + MACD/RSI confirmation.

Two labeling strategies:
    1. Triple Barrier (primary): profit target, stop loss, time barrier
    2. MACD+RSI Hybrid (secondary): momentum-based confirmation labels

The MACD+RSI hybrid generates more balanced labels than pure triple barrier
because it labels based on indicator agreement rather than price-only barriers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class LabelConfig:
    """Labeling configuration."""

    method: str = "triple_barrier"
    profit_target_pips: float = 300
    stop_loss_pips: float = 150
    max_holding_bars: int = 120
    pip_value: float = 0.10  # XAUUSD: 1 pip = $0.10
    # MACD+RSI parameters
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


class TripleBarrierLabeler:
    """Triple barrier labeling: profit target, stop loss, time barrier.

    Labels each bar based on which barrier is hit first:
        - Upper barrier (profit target): label = 2 (buy)
        - Lower barrier (stop loss): label = 0 (sell)
        - Time barrier (max holding): label = 1 (hold)
    """

    def __init__(self, cfg: LabelConfig):
        self.profit_target = cfg.profit_target_pips * cfg.pip_value
        self.stop_loss = cfg.stop_loss_pips * cfg.pip_value
        self.max_holding = cfg.max_holding_bars

    def label(self, close: np.ndarray) -> np.ndarray:
        n = len(close)
        labels = np.ones(n, dtype=np.int64)  # default: hold

        for i in range(n - 1):
            entry = close[i]
            max_j = min(i + self.max_holding, n)

            for j in range(i + 1, max_j):
                change = close[j] - entry
                if change >= self.profit_target:
                    labels[i] = 2  # buy
                    break
                elif change <= -self.stop_loss:
                    labels[i] = 0  # sell
                    break

        return labels


class MACDRSILabeler:
    """MACD + RSI hybrid labeling for more balanced class distribution.

    Generates labels based on indicator agreement:
        - Buy (2): RSI < oversold AND MACD histogram crossing up
        - Sell (0): RSI > overbought AND MACD histogram crossing down
        - Hold (1): indicators disagree or neutral

    Produces naturally balanced labels because the conditions are symmetric
    and most bars have neutral/disagreeing indicators → hold.
    """

    def __init__(self, cfg: LabelConfig):
        self.rsi_oversold = cfg.rsi_oversold
        self.rsi_overbought = cfg.rsi_overbought
        self.rsi_period = cfg.rsi_period
        self.macd_fast = cfg.macd_fast
        self.macd_slow = cfg.macd_slow
        self.macd_signal = cfg.macd_signal

    def _compute_rsi(self, close: np.ndarray) -> np.ndarray:
        deltas = np.diff(close, prepend=close[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)

        avg_gain[self.rsi_period] = gains[1 : self.rsi_period + 1].mean()
        avg_loss[self.rsi_period] = losses[1 : self.rsi_period + 1].mean()

        for i in range(self.rsi_period + 1, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * (self.rsi_period - 1) + gains[i]) / self.rsi_period
            avg_loss[i] = (avg_loss[i - 1] * (self.rsi_period - 1) + losses[i]) / self.rsi_period

        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        def ema(data, period):
            result = np.zeros_like(data)
            result[0] = data[0]
            alpha = 2 / (period + 1)
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result

        ema_fast = ema(close, self.macd_fast)
        ema_slow = ema(close, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, self.macd_signal)
        histogram = macd_line - signal_line
        return macd_line, histogram

    def label(self, close: np.ndarray) -> np.ndarray:
        rsi = self._compute_rsi(close)
        _, macd_hist = self._compute_macd(close)

        n = len(close)
        labels = np.ones(n, dtype=np.int64)  # default: hold

        warmup = max(self.rsi_period, self.macd_slow) + self.macd_signal + 5

        for i in range(warmup, n - 1):
            hist_cross_up = macd_hist[i] > 0 and macd_hist[i - 1] <= 0
            hist_cross_down = macd_hist[i] < 0 and macd_hist[i - 1] >= 0

            if rsi[i] < self.rsi_oversold and hist_cross_up:
                labels[i] = 2  # buy
            elif rsi[i] > self.rsi_overbought and hist_cross_down:
                labels[i] = 0  # sell
            # Additional: strong momentum signals
            elif rsi[i] < 25 and macd_hist[i] > 0:
                labels[i] = 2  # buy (very oversold + bullish momentum)
            elif rsi[i] > 75 and macd_hist[i] < 0:
                labels[i] = 0  # sell (very overbought + bearish momentum)

        return labels


class HybridLabeler:
    """Combines triple barrier and MACD/RSI labels with voting.

    If both methods agree → use that label.
    If they disagree → default to hold (conservative).

    This produces the most balanced and reliable labels because:
    - Triple barrier handles clear directional moves
    - MACD/RSI handles momentum-based entries
    - Disagreement → hold (reduces noise)
    """

    def __init__(self, cfg: LabelConfig):
        self.tb = TripleBarrierLabeler(cfg)
        self.mr = MACDRSILabeler(cfg)

    def label(self, close: np.ndarray) -> np.ndarray:
        tb_labels = self.tb.label(close)
        mr_labels = self.mr.label(close)

        n = len(close)
        labels = np.ones(n, dtype=np.int64)  # default: hold

        for i in range(n):
            if tb_labels[i] == mr_labels[i]:
                labels[i] = tb_labels[i]  # agreement → use it
            elif tb_labels[i] != 1 and mr_labels[i] == 1:
                labels[i] = tb_labels[i]  # TB has signal, MACD neutral → use TB
            elif mr_labels[i] != 1 and tb_labels[i] == 1:
                labels[i] = mr_labels[i]  # MACD has signal, TB neutral → use MACD
            # else: disagreement → hold (default)

        return labels


def get_labeler(cfg: LabelConfig):
    """Factory for labelers."""
    if cfg.method == "triple_barrier":
        return TripleBarrierLabeler(cfg)
    elif cfg.method == "macd_rsi":
        return MACDRSILabeler(cfg)
    elif cfg.method == "hybrid":
        return HybridLabeler(cfg)
    else:
        raise ValueError(f"Unknown labeling method: {cfg.method}")


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_length: int = 120,
    sentiment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Create overlapping sequences for time-series models.

    Returns:
        X: (n_seq, seq_length, n_features)
        y: (n_seq,)
        S: (n_seq, 768) or None — sentiment embeddings aligned to sequences
    """
    n = len(features)
    if n <= seq_length:
        raise ValueError(f"Not enough data ({n}) for seq_length {seq_length}")

    n_seq = n - seq_length
    X = np.zeros((n_seq, seq_length, features.shape[1]), dtype=np.float32)
    y = np.zeros(n_seq, dtype=np.int64)

    for i in range(n_seq):
        X[i] = features[i : i + seq_length]
        y[i] = labels[i + seq_length - 1]

    S = None
    if sentiment is not None:
        S = sentiment[seq_length - 1 : seq_length - 1 + n_seq].astype(np.float32)

    logger.info(f"Created {n_seq} sequences (length {seq_length})")
    return X, y, S
