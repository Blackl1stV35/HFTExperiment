"""Uncertainty estimation for the 'self-aware' exit strategy.

Three layers of uncertainty that independently trigger position exits:
1. Model uncertainty (MC Dropout / ensemble disagreement)
2. Regime mismatch (classifier confidence drop)
3. Data distribution shift (OOD detection)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class UncertaintySignals:
    """Aggregated uncertainty signals from all sources."""

    model_uncertainty: float = 0.0        # MC Dropout / ensemble std
    regime_confidence: float = 1.0         # regime classifier confidence
    distribution_shift: float = 0.0        # OOD score
    should_exit: bool = False
    should_reduce: bool = False
    exit_reason: str = ""

    @property
    def overall_risk_score(self) -> float:
        """Combined risk score (0 = safe, 1 = maximum risk)."""
        return max(
            self.model_uncertainty,
            1.0 - self.regime_confidence,
            self.distribution_shift,
        )


class UncertaintyMonitor:
    """Monitors model uncertainty and triggers exits when confidence drops."""

    def __init__(
        self,
        model_uncertainty_threshold: float = 0.15,
        regime_confidence_threshold: float = 0.6,
        ood_threshold: float = 3.0,
    ):
        self.model_threshold = model_uncertainty_threshold
        self.regime_threshold = regime_confidence_threshold
        self.ood_threshold = ood_threshold

        # Calibration: running statistics for OOD detection
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._calibrated = False

    def calibrate(self, training_features: np.ndarray) -> None:
        """Calibrate OOD detector using training data statistics."""
        self._feature_means = training_features.mean(axis=0)
        self._feature_stds = training_features.std(axis=0) + 1e-8
        self._calibrated = True
        logger.info(f"OOD detector calibrated on {len(training_features)} samples")

    def assess(
        self,
        model_uncertainty: float,
        regime_confidence: float,
        current_features: Optional[np.ndarray] = None,
    ) -> UncertaintySignals:
        """Assess all uncertainty sources and generate signals."""
        signals = UncertaintySignals(
            model_uncertainty=model_uncertainty,
            regime_confidence=regime_confidence,
        )

        # OOD detection via Mahalanobis-like distance
        if current_features is not None and self._calibrated:
            z_scores = np.abs(
                (current_features - self._feature_means) / self._feature_stds
            )
            signals.distribution_shift = float(z_scores.mean())

        # Decision logic
        reasons = []

        if model_uncertainty > self.model_threshold:
            signals.should_reduce = True
            reasons.append(f"model_uncertainty={model_uncertainty:.3f}")

        if regime_confidence < self.regime_threshold:
            signals.should_reduce = True
            reasons.append(f"regime_confidence={regime_confidence:.3f}")

        if signals.distribution_shift > self.ood_threshold:
            signals.should_exit = True
            reasons.append(f"OOD_score={signals.distribution_shift:.3f}")

        # If multiple signals fire simultaneously, escalate to exit
        n_warnings = sum([
            model_uncertainty > self.model_threshold,
            regime_confidence < self.regime_threshold,
            signals.distribution_shift > self.ood_threshold * 0.7,
        ])
        if n_warnings >= 2:
            signals.should_exit = True
            reasons.append("multiple_uncertainty_signals")

        if reasons:
            signals.exit_reason = "; ".join(reasons)
            level = "EXIT" if signals.should_exit else "REDUCE"
            logger.warning(f"Uncertainty {level}: {signals.exit_reason}")

        return signals
