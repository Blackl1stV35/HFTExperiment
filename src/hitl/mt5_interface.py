"""Human-in-the-loop interface v2 — richer context for mid-hold reviews.

PATCH v2 changes (Strategy 5b):
    - SignalContext extended with DrawdownContext dataclass
    - _console_approval() renders DrawdownContext block when present
    - Mid-hold review layout: PnL hero + drawdown stats + consecutive losses
    - Auto-approve threshold tightened from 0.7→0.72 (higher bar for auto-pass)
    - check_exit() no longer auto-approves ALL profitable exits — exits in
      consecutive-loss streaks still get reviewed to catch regime breaks.

Layout priority (top to bottom):
    1. Unrealized PnL (largest, highlighted)
    2. Pips from entry
    3. Agent recommendation (one-line)
    4. Risk % of account  +  drawdown context (if mid-hold review)
    5. Position details (price, lots, hold time)
    6. Confidence bar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger


PIP_VALUE = 0.10


@dataclass
class DrawdownContext:
    """Rich drawdown state passed to HITL for mid-hold reviews (Strategy 5b)."""

    consecutive_losses: int = 0          # current losing streak
    daily_pnl_usd: float = 0.0           # today's running PnL
    session_volatility_pips: float = 0.0 # ATR proxy for current session
    peak_balance: float = 10_000.0
    current_balance: float = 10_000.0

    @property
    def account_drawdown_pct(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance * 100.0


@dataclass
class SignalContext:
    """Full context for human approval decisions."""

    action: str
    confidence: float
    current_price: float
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    hold_time_bars: int = 0
    exit_reason: str = ""
    position_size_lots: float = 0.01
    account_balance: float = 10_000.0

    pips_from_entry: float = 0.0
    risk_pct: float = 0.0

    # Strategy 5b: optional rich drawdown context
    drawdown_ctx: Optional[DrawdownContext] = None

    # Optional technicals
    rsi: float = 0.0
    atr: float = 0.0
    sentiment_score: float = 0.0
    regime: str = ""

    def __post_init__(self):
        if self.entry_price > 0 and self.pips_from_entry == 0:
            price_diff = self.current_price - self.entry_price
            if "CLOSE_LONG" in self.action or (
                "CLOSE" in self.action and self.unrealized_pnl >= 0
            ):
                self.pips_from_entry = price_diff / PIP_VALUE
            elif "CLOSE_SHORT" in self.action or (
                "CLOSE" in self.action and self.unrealized_pnl < 0
            ):
                self.pips_from_entry = -price_diff / PIP_VALUE
            else:
                self.pips_from_entry = price_diff / PIP_VALUE
        if self.account_balance > 0 and self.risk_pct == 0:
            self.risk_pct = abs(self.unrealized_pnl) / self.account_balance * 100

    @property
    def recommendation(self) -> str:
        if "mid_hold_drawdown" in self.exit_reason:
            if self.drawdown_ctx and self.drawdown_ctx.consecutive_losses >= 3:
                return "CAUTION — possible regime break, consecutive losses detected"
            return f"Mid-hold review — loss at ${self.unrealized_pnl:.0f}, consider cutting"
        elif "max_hold" in self.exit_reason or "max_time" in self.exit_reason:
            return "Time-based close — position exceeded max hold period"
        elif "signal_reverse" in self.exit_reason:
            if self.unrealized_pnl > 0:
                return "Take profit — model sees reversal, locking in gains"
            elif self.unrealized_pnl < -5:
                return "Cut loss — model sees reversal, limiting damage"
            else:
                return "Signal reversal — model direction changed"
        elif "circuit" in self.exit_reason.lower() or "drawdown" in self.exit_reason.lower():
            return "RISK EXIT — circuit breaker triggered"
        elif self.action in ("BUY", "SELL"):
            if self.confidence >= 0.72:
                return f"Strong {self.action.lower()} signal — high confidence"
            elif self.confidence >= 0.5:
                return f"Moderate {self.action.lower()} signal"
            else:
                return f"Weak {self.action.lower()} signal — consider skipping"
        elif "end_of_data" in self.exit_reason:
            return "End of session — closing all positions"
        return "Agent suggests this action"


class HITLGate:
    """Human approval gate with auto-approve rules.

    PATCH v2 (Strategy 5):
        - Auto-approve threshold raised to 0.72 (was 0.70)
        - Profitable exits in consecutive-loss streaks (>=3) are reviewed
          even if in profit — detects regime breaks early
        - check_exit() receives DrawdownContext when available

    Auto-approve (no prompt):
        - High-confidence (>=0.72) small entries
        - Profitable exits with no active losing streak

    Requires human approval:
        - Any entry below 0.72 confidence
        - Large position entries
        - Any losing exit
        - Forced exits (max hold, circuit breaker)
        - Profitable exits during consecutive-loss streak >=3
        - Mid-hold drawdown reviews (new in v2)
    """

    def __init__(
        self,
        enabled: bool = True,
        auto_approve_confidence: float = 0.72,    # PATCH: was 0.70
        auto_approve_max_lots: float = 0.03,
        auto_approve_loss_streak_cap: int = 3,    # PATCH: review profits during streaks
        show_technicals: bool = False,
        approval_fn: Optional[Callable[[SignalContext], bool]] = None,
    ):
        self.enabled = enabled
        self.auto_approve_confidence = auto_approve_confidence
        self.auto_approve_max_lots = auto_approve_max_lots
        self.auto_approve_loss_streak_cap = auto_approve_loss_streak_cap
        self.show_technicals = show_technicals
        self._approval_fn = approval_fn or self._console_approval

        self.stats = {
            "approved": 0, "vetoed": 0, "auto_approved": 0, "total_prompted": 0,
        }
        self._consecutive_losses = 0  # internal streak tracker

    def record_trade_result(self, pnl_usd: float) -> None:
        """Call after each closed trade to maintain streak state."""
        if pnl_usd <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def check_entry(self, ctx: SignalContext) -> bool:
        if not self.enabled:
            self.stats["auto_approved"] += 1
            return True
        if (
            ctx.confidence >= self.auto_approve_confidence
            and ctx.position_size_lots <= self.auto_approve_max_lots
        ):
            self.stats["auto_approved"] += 1
            return True
        return self._request(ctx)

    def check_exit(self, ctx: SignalContext) -> bool:
        if not self.enabled:
            self.stats["auto_approved"] += 1
            return True

        # Auto-approve profitable exits UNLESS in a losing streak
        if (
            ctx.unrealized_pnl > 0
            and self._consecutive_losses < self.auto_approve_loss_streak_cap
        ):
            self.stats["auto_approved"] += 1
            return True

        return self._request(ctx)

    def _request(self, ctx: SignalContext) -> bool:
        self.stats["total_prompted"] += 1
        result = self._approval_fn(ctx)
        self.stats["approved" if result else "vetoed"] += 1
        return result

    def _console_approval(self, ctx: SignalContext) -> bool:
        """Console prompt — PnL-first layout with optional DrawdownContext block."""
        is_exit = "CLOSE" in ctx.action or ctx.entry_price > 0
        is_mid_hold = "mid_hold" in ctx.exit_reason

        print(f"\n{'━'*60}")

        if is_exit:
            pnl_sign = "+" if ctx.unrealized_pnl >= 0 else ""
            pnl_color = "32" if ctx.unrealized_pnl >= 0 else "31"
            print(
                f"  \033[1;{pnl_color}m  PnL: {pnl_sign}${ctx.unrealized_pnl:.2f}  "
                f"({pnl_sign}{ctx.pips_from_entry:.0f} pips)\033[0m"
            )
            print(f"  {'─'*56}")
            risk_label = "RISK" if ctx.unrealized_pnl < 0 else "GAIN"
            print(f"  {risk_label}: {ctx.risk_pct:.1f}% of account")

            # Strategy 5b: DrawdownContext block for mid-hold reviews
            if is_mid_hold and ctx.drawdown_ctx:
                dc = ctx.drawdown_ctx
                print(f"  {'─'*56}")
                print(f"  DRAWDOWN CONTEXT")
                print(f"  Account DD:       {dc.account_drawdown_pct:.1f}%")
                print(f"  Consecutive L:    {dc.consecutive_losses}")
                print(f"  Daily PnL:        ${dc.daily_pnl_usd:.2f}")
                if dc.session_volatility_pips > 0:
                    print(f"  Session ATR:      {dc.session_volatility_pips:.1f} pips")
        else:
            print(f"  NEW {ctx.action}  |  Lots: {ctx.position_size_lots}")
            print(f"  {'─'*56}")

        print(f"  >> {ctx.recommendation}")
        print(f"  {'─'*56}")

        if is_exit:
            print(f"  Action:     {ctx.action}")
            print(f"  Entry:      {ctx.entry_price:.2f}  →  Now: {ctx.current_price:.2f}")
            print(f"  Lots:       {ctx.position_size_lots}")
            print(f"  Hold:       {ctx.hold_time_bars} bars ({ctx.hold_time_bars} min)")
            print(f"  Reason:     {ctx.exit_reason}")
        else:
            print(f"  Price:      {ctx.current_price:.2f}")

        conf_filled = int(ctx.confidence * 20)
        conf_bar = "█" * conf_filled + "░" * (20 - conf_filled)
        print(f"  Confidence: [{conf_bar}] {ctx.confidence:.1%}")

        if self.show_technicals:
            parts = []
            if ctx.rsi > 0: parts.append(f"RSI:{ctx.rsi:.0f}")
            if ctx.atr > 0: parts.append(f"ATR:{ctx.atr:.2f}")
            if ctx.sentiment_score != 0: parts.append(f"Sent:{ctx.sentiment_score:.3f}")
            if ctx.regime: parts.append(f"Regime:{ctx.regime}")
            if parts:
                print(f"  Tech: {' | '.join(parts)}")

        print(f"{'━'*60}")

        while True:
            resp = input("  Approve? (y/n/s=skip all): ").strip().lower()
            if resp in ("y", "yes"):
                return True
            if resp in ("n", "no"):
                print("  → VETOED")
                return False
            if resp in ("s", "skip"):
                self.enabled = False
                print("  → Auto-approving all remaining")
                return True


class RiskDisplay:
    """Compact signal display for periodic logging."""

    @staticmethod
    def format_signal(action: int, confidence: float, price: float) -> str:
        action_map = {0: "SELL ↓", 1: "HOLD ─", 2: "BUY ↑"}
        conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        return f"{action_map.get(action, '???')} [{conf_bar}] {confidence:.0%} @ {price:.2f}"
