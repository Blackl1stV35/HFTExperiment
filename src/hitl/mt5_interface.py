"""Human-in-the-loop MT5 interface: signal display + approval gate.

The autonomous model suggests entries and exits, but:
    - Large positions require human approval
    - Drawdown-triggered exits require human approval
    - All exits display explainable risk metrics before asking

This module provides both console-based prompts (for backtest/paper trade)
and structured output for a future GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from loguru import logger


@dataclass
class SignalContext:
    """Context displayed to the human operator for approval decisions."""

    action: str             # "BUY", "SELL", "CLOSE_LONG", "CLOSE_SHORT"
    confidence: float       # model confidence [0, 1]
    current_price: float
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    hold_time_bars: int = 0
    exit_reason: str = ""
    regime: str = "unknown"
    # Explainable features
    rsi: float = 50.0
    atr: float = 0.0
    sentiment_score: float = 0.0
    position_size_lots: float = 0.01


class HITLGate:
    """Human-in-the-loop approval gate for trading decisions.

    Rules:
        - Entries with confidence > threshold: auto-approved (small size)
        - Entries with large size: requires approval
        - All exits on losing positions: requires approval
        - Circuit breaker exits: requires approval
        - Hold signals: auto-approved (no action needed)
    """

    def __init__(
        self,
        enabled: bool = True,
        auto_approve_confidence: float = 0.7,
        auto_approve_max_lots: float = 0.03,
        approval_fn: Optional[Callable[[SignalContext], bool]] = None,
    ):
        self.enabled = enabled
        self.auto_approve_confidence = auto_approve_confidence
        self.auto_approve_max_lots = auto_approve_max_lots
        self._approval_fn = approval_fn or self._console_approval

        self.stats = {"approved": 0, "vetoed": 0, "auto_approved": 0}

    def check_entry(self, ctx: SignalContext) -> bool:
        """Check if a new entry is approved."""
        if not self.enabled:
            self.stats["auto_approved"] += 1
            return True

        # Auto-approve high-confidence small entries
        if (
            ctx.confidence >= self.auto_approve_confidence
            and ctx.position_size_lots <= self.auto_approve_max_lots
        ):
            self.stats["auto_approved"] += 1
            return True

        # Requires human approval
        return self._request_approval(ctx, "ENTRY")

    def check_exit(self, ctx: SignalContext) -> bool:
        """Check if an exit is approved.

        Profitable exits are auto-approved.
        Losing exits and forced exits require approval.
        """
        if not self.enabled:
            self.stats["auto_approved"] += 1
            return True

        # Auto-approve profitable exits
        if ctx.unrealized_pnl > 0:
            self.stats["auto_approved"] += 1
            return True

        # Losing or forced exits require approval
        return self._request_approval(ctx, "EXIT")

    def _request_approval(self, ctx: SignalContext, action_type: str) -> bool:
        result = self._approval_fn(ctx)
        if result:
            self.stats["approved"] += 1
        else:
            self.stats["vetoed"] += 1
        return result

    @staticmethod
    def _console_approval(ctx: SignalContext) -> bool:
        """Console-based approval prompt."""
        print(f"\n{'='*55}")
        print(f"  APPROVAL REQUIRED: {ctx.action}")
        print(f"  {'─'*51}")
        print(f"  Price:      {ctx.current_price:.2f}")
        if ctx.entry_price > 0:
            print(f"  Entry:      {ctx.entry_price:.2f}")
            print(f"  Unreal PnL: ${ctx.unrealized_pnl:.2f}")
            print(f"  Hold time:  {ctx.hold_time_bars} bars")
        print(f"  Confidence: {ctx.confidence:.1%}")
        print(f"  Size:       {ctx.position_size_lots} lots")
        print(f"  Regime:     {ctx.regime}")
        if ctx.exit_reason:
            print(f"  Reason:     {ctx.exit_reason}")
        print(f"  {'─'*51}")
        print(f"  RSI: {ctx.rsi:.1f}  ATR: {ctx.atr:.2f}  Sentiment: {ctx.sentiment_score:.3f}")
        print(f"{'='*55}")

        while True:
            resp = input("  Approve? (y/n): ").strip().lower()
            if resp in ("y", "yes"):
                return True
            if resp in ("n", "no"):
                print("  → VETOED")
                return False


class RiskDisplay:
    """Explainable risk metrics display for the human operator."""

    @staticmethod
    def format_signal(
        action: int,
        confidence: float,
        price: float,
        regime: str = "unknown",
        rsi: float = 50.0,
        atr: float = 0.0,
    ) -> str:
        """Format a signal for display."""
        action_map = {0: "SELL ↓", 1: "HOLD ─", 2: "BUY ↑"}
        conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))

        return (
            f"Signal: {action_map.get(action, '???')} | "
            f"Conf: [{conf_bar}] {confidence:.1%} | "
            f"Price: {price:.2f} | "
            f"Regime: {regime} | "
            f"RSI: {rsi:.0f} | ATR: {atr:.2f}"
        )
