"""Human-in-the-loop interface — redesigned for practical use.

Layout priority (top to bottom):
    1. Unrealized PnL (largest, highlighted)
    2. Pips from entry
    3. Agent recommendation (one-line)
    4. Risk % of account
    5. Position details (price, lots, hold time)
    6. Confidence bar

Removed: RSI, ATR, Sentiment (noise when sentiment=0, RSI/ATR not in the RL obs).
These can be re-enabled via show_technicals=True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger


PIP_VALUE = 0.10  # XAUUSD: 1 pip = $0.10 price move


@dataclass
class SignalContext:
    """Full context for human approval decisions."""

    action: str                     # "BUY", "SELL", "CLOSE_LONG", "CLOSE_SHORT"
    confidence: float               # model confidence [0, 1]
    current_price: float
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0     # USD
    hold_time_bars: int = 0
    exit_reason: str = ""
    position_size_lots: float = 0.01
    account_balance: float = 10_000.0

    # Computed fields (filled by the caller or auto-computed)
    pips_from_entry: float = 0.0    # signed: positive = in profit direction
    risk_pct: float = 0.0           # |unrealized_pnl| / account_balance × 100

    # Optional technicals (only shown if show_technicals=True)
    rsi: float = 0.0
    atr: float = 0.0
    sentiment_score: float = 0.0
    regime: str = ""

    def __post_init__(self):
        """Auto-compute derived fields if not explicitly set."""
        if self.entry_price > 0 and self.pips_from_entry == 0:
            price_diff = self.current_price - self.entry_price
            # For exits, sign indicates profit direction
            if "CLOSE_LONG" in self.action or ("CLOSE" in self.action and self.unrealized_pnl >= 0):
                self.pips_from_entry = price_diff / PIP_VALUE
            elif "CLOSE_SHORT" in self.action or ("CLOSE" in self.action and self.unrealized_pnl < 0):
                self.pips_from_entry = -price_diff / PIP_VALUE
            else:
                self.pips_from_entry = price_diff / PIP_VALUE

        if self.account_balance > 0 and self.risk_pct == 0:
            self.risk_pct = abs(self.unrealized_pnl) / self.account_balance * 100

    @property
    def recommendation(self) -> str:
        """One-line agent recommendation based on context."""
        if "max_hold" in self.exit_reason or "max_time" in self.exit_reason:
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
            if self.confidence >= 0.7:
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

    Auto-approve (no prompt):
        - High-confidence small entries
        - Profitable exits
    
    Requires human approval:
        - Low-confidence entries
        - Large position entries
        - Losing exits
        - Forced exits (max hold, circuit breaker)
    """

    def __init__(
        self,
        enabled: bool = True,
        auto_approve_confidence: float = 0.7,
        auto_approve_max_lots: float = 0.03,
        show_technicals: bool = False,
        approval_fn: Optional[Callable[[SignalContext], bool]] = None,
    ):
        self.enabled = enabled
        self.auto_approve_confidence = auto_approve_confidence
        self.auto_approve_max_lots = auto_approve_max_lots
        self.show_technicals = show_technicals
        self._approval_fn = approval_fn or self._console_approval

        self.stats = {
            "approved": 0, "vetoed": 0, "auto_approved": 0,
            "total_prompted": 0,
        }

    def check_entry(self, ctx: SignalContext) -> bool:
        if not self.enabled:
            self.stats["auto_approved"] += 1
            return True
        if (ctx.confidence >= self.auto_approve_confidence
                and ctx.position_size_lots <= self.auto_approve_max_lots):
            self.stats["auto_approved"] += 1
            return True
        return self._request(ctx)

    def check_exit(self, ctx: SignalContext) -> bool:
        if not self.enabled:
            self.stats["auto_approved"] += 1
            return True
        if ctx.unrealized_pnl > 0:
            self.stats["auto_approved"] += 1
            return True
        return self._request(ctx)

    def _request(self, ctx: SignalContext) -> bool:
        self.stats["total_prompted"] += 1
        result = self._approval_fn(ctx)
        self.stats["approved" if result else "vetoed"] += 1
        return result

    def _console_approval(self, ctx: SignalContext) -> bool:
        """Redesigned console prompt — PnL-first layout."""
        is_exit = "CLOSE" in ctx.action or ctx.entry_price > 0

        # ── Header ──
        print(f"\n{'━'*58}")
        if is_exit:
            pnl_sign = "+" if ctx.unrealized_pnl >= 0 else ""
            pnl_color = "32" if ctx.unrealized_pnl >= 0 else "31"  # green/red ANSI

            # PnL is the HERO element
            print(f"  \033[1;{pnl_color}m  PnL: {pnl_sign}${ctx.unrealized_pnl:.2f}  "
                  f"({pnl_sign}{ctx.pips_from_entry:.0f} pips)\033[0m")
            print(f"  {'─'*54}")

            # Risk percentage
            risk_label = "RISK" if ctx.unrealized_pnl < 0 else "GAIN"
            print(f"  {risk_label}: {ctx.risk_pct:.1f}% of account")
        else:
            print(f"  NEW {ctx.action}  |  Lots: {ctx.position_size_lots}")
            print(f"  {'─'*54}")

        # ── Agent recommendation ──
        print(f"  >> {ctx.recommendation}")
        print(f"  {'─'*54}")

        # ── Position details ──
        if is_exit:
            print(f"  Action:     {ctx.action}")
            print(f"  Entry:      {ctx.entry_price:.2f}  →  Now: {ctx.current_price:.2f}")
            print(f"  Lots:       {ctx.position_size_lots}")
            print(f"  Hold:       {ctx.hold_time_bars} bars ({ctx.hold_time_bars} min)")
            print(f"  Reason:     {ctx.exit_reason}")
        else:
            print(f"  Price:      {ctx.current_price:.2f}")

        # ── Confidence bar ──
        conf_filled = int(ctx.confidence * 20)
        conf_bar = "█" * conf_filled + "░" * (20 - conf_filled)
        print(f"  Confidence: [{conf_bar}] {ctx.confidence:.1%}")

        # ── Optional technicals ──
        if self.show_technicals:
            parts = []
            if ctx.rsi > 0:
                parts.append(f"RSI:{ctx.rsi:.0f}")
            if ctx.atr > 0:
                parts.append(f"ATR:{ctx.atr:.2f}")
            if ctx.sentiment_score != 0:
                parts.append(f"Sent:{ctx.sentiment_score:.3f}")
            if ctx.regime:
                parts.append(f"Regime:{ctx.regime}")
            if parts:
                print(f"  Tech: {' | '.join(parts)}")

        print(f"{'━'*58}")

        # ── Decision ──
        while True:
            resp = input("  Approve? (y/n/s=skip all): ").strip().lower()
            if resp in ("y", "yes"):
                return True
            if resp in ("n", "no"):
                print("  → VETOED")
                return False
            if resp in ("s", "skip"):
                # Auto-approve everything remaining
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
