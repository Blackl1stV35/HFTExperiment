"""Circuit breakers and risk management — the NON-NEGOTIABLE safety layer.

These are hard limits that override ALL model predictions. No ML model
should ever be able to bypass these controls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger


@dataclass
class RiskState:
    """Current risk state tracking."""

    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    last_trade_time: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""
    position_size_multiplier: float = 1.0
    day_start: str = ""


class CircuitBreaker:
    """Hard safety limits that override all model decisions.

    These are NOT configurable at runtime — they can only be changed
    by modifying the deployment config and restarting.
    """

    def __init__(
        self,
        max_daily_drawdown_pct: float = 2.0,
        max_position_risk_pct: float = 2.0,
        max_consecutive_losses: int = 5,
        loss_reduction_factor: float = 0.5,
        recovery_trades: int = 10,
        latency_kill_ms: float = 50.0,
        news_blackout_minutes: int = 30,
        account_balance: float = 10_000.0,
    ):
        self.max_daily_drawdown = max_daily_drawdown_pct / 100.0 * account_balance
        self.max_position_risk_pct = max_position_risk_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.loss_reduction_factor = loss_reduction_factor
        self.recovery_trades = recovery_trades
        self.latency_kill_ms = latency_kill_ms
        self.news_blackout_minutes = news_blackout_minutes
        self.account_balance = account_balance

        self.state = RiskState()
        self._news_blackout_times: list[datetime] = []

    def check_can_trade(self, inference_latency_ms: float = 0.0) -> tuple[bool, str]:
        """Check ALL circuit breakers. Returns (can_trade, reason_if_blocked)."""

        # 1. Daily drawdown check
        if abs(self.state.daily_pnl) >= self.max_daily_drawdown and self.state.daily_pnl < 0:
            self.state.is_halted = True
            self.state.halt_reason = (
                f"Daily drawdown limit hit: ${self.state.daily_pnl:.2f}"
            )
            logger.error(f"CIRCUIT BREAKER: {self.state.halt_reason}")
            return False, self.state.halt_reason

        # 2. Consecutive losses
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            self.state.position_size_multiplier = self.loss_reduction_factor
            logger.warning(
                f"Consecutive loss limit ({self.max_consecutive_losses}) — "
                f"reducing position size by {self.loss_reduction_factor}"
            )
            # Don't halt, but reduce size

        # 3. Latency check
        if inference_latency_ms > self.latency_kill_ms:
            logger.warning(
                f"Latency spike: {inference_latency_ms:.1f}ms > {self.latency_kill_ms}ms — skipping"
            )
            return False, f"Latency too high: {inference_latency_ms:.1f}ms"

        # 4. News blackout
        if self._in_news_blackout():
            return False, "News blackout period active"

        # 5. Halt state
        if self.state.is_halted:
            return False, self.state.halt_reason

        return True, ""

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade result."""
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        self.state.last_trade_time = time.time()

        if pnl <= 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
            # Gradually restore position size after recovery
            if self.state.position_size_multiplier < 1.0:
                self.state.position_size_multiplier = min(
                    1.0,
                    self.state.position_size_multiplier + (1.0 / self.recovery_trades),
                )

    def get_position_size(self, base_lots: float = 0.01) -> float:
        """Get risk-adjusted position size."""
        return base_lots * self.state.position_size_multiplier

    def add_news_event(self, event_time: datetime) -> None:
        """Register a scheduled news event for blackout enforcement."""
        self._news_blackout_times.append(event_time)

    def _in_news_blackout(self) -> bool:
        """Check if we're within a news blackout window."""
        now = datetime.utcnow()
        blackout = timedelta(minutes=self.news_blackout_minutes)

        for event_time in self._news_blackout_times:
            if abs((now - event_time).total_seconds()) < blackout.total_seconds():
                return True
        return False

    def reset_daily(self) -> None:
        """Reset daily counters (call at market open)."""
        logger.info(
            f"Daily reset | Previous day PnL: ${self.state.daily_pnl:.2f} | "
            f"Trades: {self.state.daily_trades}"
        )
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.is_halted = False
        self.state.halt_reason = ""
        self.state.day_start = datetime.utcnow().strftime("%Y-%m-%d")
        self._news_blackout_times = []


class PositionSizer:
    """Calculate position size based on account risk and volatility."""

    def __init__(
        self,
        account_balance: float = 10_000.0,
        max_risk_pct: float = 2.0,
        pip_value: float = 0.01,
    ):
        self.account_balance = account_balance
        self.max_risk_pct = max_risk_pct
        self.pip_value = pip_value

    def calculate(
        self,
        stop_loss_pips: float,
        volatility_multiplier: float = 1.0,
        circuit_breaker_multiplier: float = 1.0,
    ) -> float:
        """Calculate position size in lots.

        Args:
            stop_loss_pips: Distance to stop loss in pips.
            volatility_multiplier: Reduce size in high volatility (0-1).
            circuit_breaker_multiplier: From CircuitBreaker (0-1).

        Returns:
            Position size in lots (minimum 0.01).
        """
        risk_amount = self.account_balance * (self.max_risk_pct / 100.0)
        risk_per_pip = risk_amount / max(stop_loss_pips, 1.0)

        # Convert to lots (1 lot XAUUSD ≈ $1 per pip for 0.01)
        lots = risk_per_pip / 100.0  # $100 per pip per standard lot

        # Apply multipliers
        lots *= volatility_multiplier
        lots *= circuit_breaker_multiplier

        # Clamp to valid range
        lots = max(0.01, round(lots, 2))
        lots = min(lots, 1.0)  # hard max

        return lots

    def update_balance(self, new_balance: float) -> None:
        """Update account balance for position sizing."""
        self.account_balance = new_balance
