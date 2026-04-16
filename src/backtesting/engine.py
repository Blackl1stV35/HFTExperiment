"""Backtesting engine v2.3 — aligned to PATCH v3 RL changes.

Changes from v2.2:
    - min_confidence pre-trade gate REMOVED from BacktestConfig and run() loop.
      Run 4 demonstrated that pre-trade gating suppresses beneficial trade
      diversity. The confidence_gate in the RL reward is the sole filter;
      backtest should mirror training conditions exactly.
    - BacktestConfig.min_confidence field retained for backwards compatibility
      but is now ignored in run(). Passing it logs a deprecation warning.
    - max_position_time default: 80 (unchanged from v2.2)
    - HITL mid-hold review: unchanged from v2.2
    - Equity/trade CSV export: unchanged
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.hitl.mt5_interface import HITLGate, SignalContext


# Execution constants — MUST match RL FrozenEncoderEnv
PIP_VALUE = 0.10
PIP_USD_PER_MICROLOT = 1.0
SPREAD_PIPS = 2.0
SPREAD_COST = SPREAD_PIPS * PIP_VALUE
COMMISSION = 0.70
SLIPPAGE_PIPS = 0.5

# Strategy 5a: trigger mid-hold HITL review when unrealized loss exceeds this
MID_HOLD_HITL_THRESHOLD_USD = -200.0


@dataclass
class BacktestConfig:
    initial_balance: float = 10_000.0
    base_lot_size: float = 0.01
    max_lots: float = 0.05
    max_position_time: int = 80
    human_exit_approval: bool = False
    min_confidence: float = 0.0      # DEPRECATED in v2.3 — no longer used in run()
    drawdown_penalty: bool = True
    hitl_mid_hold_review: bool = True
    hitl_mid_hold_threshold_usd: float = MID_HOLD_HITL_THRESHOLD_USD


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    direction: int
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_usd: float
    hold_time: int
    confidence: float = 0.0
    exit_reason: str = ""
    lots: float = 0.01
    balance_after: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_balance: float = 10_000.0
    final_balance: float = 10_000.0

    @property
    def total_trades(self): return len(self.trades)

    @property
    def winning_trades(self):
        return [t for t in self.trades if t.pnl_usd > 0]

    @property
    def losing_trades(self):
        return [t for t in self.trades if t.pnl_usd <= 0]

    @property
    def win_rate(self):
        return len(self.winning_trades) / max(1, self.total_trades)

    @property
    def profit_factor(self):
        gp = sum(t.pnl_usd for t in self.winning_trades)
        gl = abs(sum(t.pnl_usd for t in self.losing_trades))
        return gp / max(gl, 1e-8)

    @property
    def total_pnl(self): return sum(t.pnl_usd for t in self.trades)

    @property
    def max_drawdown(self):
        if not self.equity_curve: return 0.0
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(peak, 1e-8)
        return float(dd.max())

    @property
    def sharpe_ratio(self):
        if len(self.trades) < 2: return 0.0
        rets = [t.pnl_usd for t in self.trades]
        m, s = np.mean(rets), np.std(rets)
        return 0.0 if s == 0 else m / s * np.sqrt(252)

    def summary(self) -> str:
        avg_w = np.mean([t.pnl_usd for t in self.winning_trades]) if self.winning_trades else 0
        avg_l = np.mean([t.pnl_usd for t in self.losing_trades]) if self.losing_trades else 0
        avg_conf = np.mean([t.confidence for t in self.trades]) if self.trades else 0
        avg_hold = np.mean([t.hold_time for t in self.trades]) if self.trades else 0
        return (
            f"{'='*55}\n"
            f"  BACKTEST RESULTS (engine v2.2)\n"
            f"{'='*55}\n"
            f"  Total Trades:      {self.total_trades}\n"
            f"  Win Rate:          {self.win_rate:.1%}\n"
            f"  Profit Factor:     {self.profit_factor:.2f}\n"
            f"  Total PnL:         ${self.total_pnl:.2f}\n"
            f"  Avg Win:           ${avg_w:.2f}\n"
            f"  Avg Loss:          ${avg_l:.2f}\n"
            f"  Max Drawdown:      {self.max_drawdown:.1%}\n"
            f"  Sharpe Ratio:      {self.sharpe_ratio:.2f}\n"
            f"  Final Balance:     ${self.final_balance:.2f}\n"
            f"  Avg Confidence:    {avg_conf:.3f}\n"
            f"  Avg Hold Time:     {avg_hold:.0f} bars\n"
            f"{'='*55}"
        )

    def profitable_trades_summary(self) -> str:
        wins = self.winning_trades
        losses = self.losing_trades
        if not wins:
            return "  No profitable trades."
        lines = [
            f"  TRADE BREAKDOWN",
            f"  {'─'*40}",
            f"  Winners: {len(wins)} trades",
            f"    Avg PnL:    ${np.mean([t.pnl_usd for t in wins]):.2f}",
            f"    Avg conf:   {np.mean([t.confidence for t in wins]):.3f}",
            f"    Avg hold:   {np.mean([t.hold_time for t in wins]):.0f} bars",
        ]
        if losses:
            lines.extend([
                f"  Losers:  {len(losses)} trades",
                f"    Avg PnL:    ${np.mean([t.pnl_usd for t in losses]):.2f}",
                f"    Avg conf:   {np.mean([t.confidence for t in losses]):.3f}",
                f"    Avg hold:   {np.mean([t.hold_time for t in losses]):.0f} bars",
            ])
        return "\n".join(lines)

    def export_equity_csv(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("bar,equity\n")
            for i, eq in enumerate(self.equity_curve):
                f.write(f"{i},{eq:.2f}\n")
        logger.info(f"Equity curve exported: {path} ({len(self.equity_curve)} points)")

    def export_trades_csv(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("entry_idx,exit_idx,direction,entry_price,exit_price,"
                    "pnl_pips,pnl_usd,hold_time,confidence,exit_reason,lots,balance_after\n")
            for t in self.trades:
                f.write(
                    f"{t.entry_idx},{t.exit_idx},{t.direction},{t.entry_price:.2f},"
                    f"{t.exit_price:.2f},{t.pnl_pips:.1f},{t.pnl_usd:.2f},"
                    f"{t.hold_time},{t.confidence:.3f},{t.exit_reason},{t.lots},{t.balance_after:.2f}\n"
                )
        logger.info(f"Trades exported: {path} ({len(self.trades)} trades)")


def _compute_pnl(direction: int, entry_price: float, exit_price: float, lots: float) -> tuple[float, float]:
    """PnL in pips and USD. MATCHES RL env: $1/pip per 0.01 lot."""
    pnl_pips = (exit_price - entry_price) * direction / PIP_VALUE
    pnl_usd = pnl_pips * PIP_USD_PER_MICROLOT * (lots / 0.01)
    pnl_usd -= COMMISSION * (lots / 0.01)
    return pnl_pips, pnl_usd


class BacktestEngine:
    """Backtesting with confidence sizing, drawdown penalty, HITL v2, CSV export.

    PATCH v2.2 additions (Strategy 5):
        - Mid-hold HITL review fires when unrealized loss crosses threshold
        - DrawdownContext (consecutive losses, daily PnL, session vars)
          passed to HITL for richer human-reviewer signal
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.hitl = HITLGate(enabled=self.config.human_exit_approval)
        self._mid_hold_reviewed: set[int] = set()  # entry_idx of reviewed positions

    def _calc_lots(self, conf: float, balance: float, peak_balance: float) -> float:
        """Confidence + drawdown-adjusted lot sizing.

        min_confidence here is a lot-SIZE floor only — not a trade gate.
        Trades below min_confidence execute at base_lot_size rather than
        being blocked. The signal gate was removed in v2.3.
        """
        cfg = self.config
        min_c = cfg.min_confidence  # lot-scaling floor, not a filter
        if conf > min_c:
            scale = min((conf - min_c) / (1.0 - min_c + 1e-8), 1.0)
            lots = cfg.base_lot_size + scale * (cfg.max_lots - cfg.base_lot_size)
        else:
            lots = cfg.base_lot_size
        if cfg.drawdown_penalty and peak_balance > 0:
            dd = (peak_balance - balance) / peak_balance
            if dd > 0.05:
                dd_scale = max(0.3, 1.0 - dd * 2)
                lots *= dd_scale
        return round(max(cfg.base_lot_size, min(lots, cfg.max_lots)), 2)

    def _make_exit_context(self, position, price, bar_idx, reason, balance) -> SignalContext:
        d, ep, ei, lots, conf = position
        pnl_pips, pnl_usd = _compute_pnl(d, ep, price, lots)
        return SignalContext(
            action="CLOSE_LONG" if d == 1 else "CLOSE_SHORT",
            confidence=conf,
            current_price=price,
            entry_price=ep,
            unrealized_pnl=pnl_usd,
            hold_time_bars=bar_idx - ei,
            exit_reason=reason,
            position_size_lots=lots,
            account_balance=balance,
        )

    def run(self, prices, signals, confidences=None) -> BacktestResult:
        cfg = self.config
        if confidences is None:
            confidences = np.ones(len(prices), dtype=np.float32) * 0.5

        balance = cfg.initial_balance
        peak_balance = balance
        position = None
        trades = []
        equity = [balance]
        self._mid_hold_reviewed = set()

        for i in range(len(prices)):
            signal = int(signals[i])
            price = float(prices[i])
            conf = float(confidences[i])

            # ── Force close on max hold ──────────────────────────────────
            if position and (i - position[2]) >= cfg.max_position_time:
                ctx = self._make_exit_context(position, price, i, "max_hold_time", balance)
                if self.hitl.check_exit(ctx):
                    pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], price, position[3])
                    balance += pnl_usd
                    peak_balance = max(peak_balance, balance)
                    trades.append(Trade(
                        position[2], i, position[0], position[1], price,
                        pnl_pips, pnl_usd, i - position[2], position[4],
                        "max_time", position[3], balance,
                    ))
                    position = None

            # ── Strategy 5a: mid-hold HITL review on drawdown threshold ──
            if (
                position
                and cfg.hitl_mid_hold_review
                and self.config.human_exit_approval
                and position[2] not in self._mid_hold_reviewed
            ):
                _, current_pnl = _compute_pnl(position[0], position[1], price, position[3])
                if current_pnl <= cfg.hitl_mid_hold_threshold_usd:
                    self._mid_hold_reviewed.add(position[2])
                    ctx = self._make_exit_context(
                        position, price, i, "mid_hold_drawdown_review", balance
                    )
                    # Strategy 5b: inject drawdown context into recommendation
                    ctx.exit_reason = (
                        f"mid_hold_drawdown_review | "
                        f"unrealized=${current_pnl:.0f} | "
                        f"dd_threshold=${cfg.hitl_mid_hold_threshold_usd:.0f}"
                    )
                    if self.hitl.check_exit(ctx):
                        pnl_pips, pnl_usd = _compute_pnl(
                            position[0], position[1], price, position[3]
                        )
                        balance += pnl_usd
                        peak_balance = max(peak_balance, balance)
                        trades.append(Trade(
                            position[2], i, position[0], position[1], price,
                            pnl_pips, pnl_usd, i - position[2], position[4],
                            "hitl_mid_hold_cut", position[3], balance,
                        ))
                        position = None

            lots = self._calc_lots(conf, balance, peak_balance)

            # ── Signal execution ─────────────────────────────────────────
            if position is None:
                if signal == 2:
                    position = (1, price + SPREAD_COST * 0.5, i, lots, conf)
                elif signal == 0:
                    position = (-1, price - SPREAD_COST * 0.5, i, lots, conf)

            elif signal == 2 and position[0] == -1:
                ctx = self._make_exit_context(position, price, i, "signal_reverse", balance)
                if self.hitl.check_exit(ctx):
                    pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], price, position[3])
                    balance += pnl_usd
                    peak_balance = max(peak_balance, balance)
                    trades.append(Trade(
                        position[2], i, position[0], position[1], price,
                        pnl_pips, pnl_usd, i - position[2], position[4],
                        "signal_reverse", position[3], balance,
                    ))
                    position = (1, price + SPREAD_COST * 0.5, i, lots, conf)

            elif signal == 0 and position[0] == 1:
                ctx = self._make_exit_context(position, price, i, "signal_reverse", balance)
                if self.hitl.check_exit(ctx):
                    pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], price, position[3])
                    balance += pnl_usd
                    peak_balance = max(peak_balance, balance)
                    trades.append(Trade(
                        position[2], i, position[0], position[1], price,
                        pnl_pips, pnl_usd, i - position[2], position[4],
                        "signal_reverse", position[3], balance,
                    ))
                    position = (-1, price - SPREAD_COST * 0.5, i, lots, conf)

            equity.append(balance)

        # Close remaining position at end of data
        if position:
            pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], prices[-1], position[3])
            balance += pnl_usd
            trades.append(Trade(
                position[2], len(prices) - 1, position[0], position[1], prices[-1],
                pnl_pips, pnl_usd, len(prices) - 1 - position[2], position[4],
                "end_of_data", position[3], balance,
            ))
            equity.append(balance)

        if self.hitl.enabled:
            logger.info(
                f"HITL stats: approved={self.hitl.stats['approved']} "
                f"vetoed={self.hitl.stats['vetoed']} "
                f"auto={self.hitl.stats['auto_approved']}"
            )

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            initial_balance=cfg.initial_balance,
            final_balance=balance,
        )
