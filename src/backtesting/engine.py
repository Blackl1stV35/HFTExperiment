"""Backtesting engine v2.1 — equity export, drawdown penalty, account-aware HITL.

Changes from v2.0:
    - Passes account_balance through SignalContext so HITL shows risk %
    - Equity curve exported as CSV alongside NPZ
    - Optional drawdown penalty in confidence sizing
    - Cleaner trade logging
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


@dataclass
class BacktestConfig:
    initial_balance: float = 10_000.0
    base_lot_size: float = 0.01
    max_lots: float = 0.05
    max_position_time: int = 120
    human_exit_approval: bool = False
    min_confidence: float = 0.0
    drawdown_penalty: bool = True  # reduce size during drawdown


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
            f"  BACKTEST RESULTS\n"
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
        """Export equity curve as CSV."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("bar,equity\n")
            for i, eq in enumerate(self.equity_curve):
                f.write(f"{i},{eq:.2f}\n")
        logger.info(f"Equity curve exported: {path} ({len(self.equity_curve)} points)")

    def export_trades_csv(self, path: str) -> None:
        """Export trade log as CSV."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("entry_idx,exit_idx,direction,entry_price,exit_price,pnl_pips,pnl_usd,hold_time,confidence,exit_reason,lots,balance_after\n")
            for t in self.trades:
                f.write(f"{t.entry_idx},{t.exit_idx},{t.direction},{t.entry_price:.2f},"
                        f"{t.exit_price:.2f},{t.pnl_pips:.1f},{t.pnl_usd:.2f},"
                        f"{t.hold_time},{t.confidence:.3f},{t.exit_reason},{t.lots},{t.balance_after:.2f}\n")
        logger.info(f"Trades exported: {path} ({len(self.trades)} trades)")


def _compute_pnl(direction: int, entry_price: float, exit_price: float, lots: float) -> tuple[float, float]:
    """PnL in pips and USD. MATCHES RL env: $1/pip per 0.01 lot."""
    pnl_pips = (exit_price - entry_price) * direction / PIP_VALUE
    pnl_usd = pnl_pips * PIP_USD_PER_MICROLOT * (lots / 0.01)
    pnl_usd -= COMMISSION * (lots / 0.01)
    return pnl_pips, pnl_usd


class BacktestEngine:
    """Backtesting with confidence sizing, drawdown penalty, HITL, CSV export."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.hitl = HITLGate(enabled=self.config.human_exit_approval)

    def _calc_lots(self, conf: float, balance: float, peak_balance: float) -> float:
        """Confidence + drawdown-adjusted lot sizing."""
        cfg = self.config
        min_c = cfg.min_confidence

        # Base: scale by confidence
        if conf > min_c:
            scale = min((conf - min_c) / (1.0 - min_c + 1e-8), 1.0)
            lots = cfg.base_lot_size + scale * (cfg.max_lots - cfg.base_lot_size)
        else:
            lots = cfg.base_lot_size

        # Drawdown penalty: reduce size when in drawdown
        if cfg.drawdown_penalty and peak_balance > 0:
            dd = (peak_balance - balance) / peak_balance
            if dd > 0.05:  # >5% drawdown
                dd_scale = max(0.3, 1.0 - dd * 2)  # at 25% DD → 50% size reduction
                lots *= dd_scale

        return round(max(cfg.base_lot_size, min(lots, cfg.max_lots)), 2)

    def _make_exit_context(self, position, price, bar_idx, reason, balance) -> SignalContext:
        """Build SignalContext for HITL with all computed fields."""
        d, ep, ei, lots, conf = position
        pnl_pips, pnl_usd = _compute_pnl(d, ep, price, lots)

        ctx = SignalContext(
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
        return ctx

    def run(self, prices, signals, confidences=None) -> BacktestResult:
        cfg = self.config
        if confidences is None:
            confidences = np.ones(len(prices), dtype=np.float32) * 0.5

        balance = cfg.initial_balance
        peak_balance = balance
        position = None
        trades = []
        equity = [balance]
        filtered_count = 0

        for i in range(len(prices)):
            signal = int(signals[i])
            price = float(prices[i])
            conf = float(confidences[i])

            # Confidence filter
            if conf < cfg.min_confidence and signal != 1:
                signal = 1
                filtered_count += 1

            # Force close on max hold
            if position and (i - position[2]) >= cfg.max_position_time:
                ctx = self._make_exit_context(position, price, i, "max_hold_time", balance)
                if self.hitl.check_exit(ctx):
                    pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], price, position[3])
                    balance += pnl_usd
                    peak_balance = max(peak_balance, balance)
                    trades.append(Trade(position[2], i, position[0], position[1], price,
                                        pnl_pips, pnl_usd, i-position[2], position[4],
                                        "max_time", position[3], balance))
                    position = None

            # Lot sizing
            lots = self._calc_lots(conf, balance, peak_balance)

            # Execute
            if signal == 2 and position is None:
                position = (1, price + SPREAD_COST * 0.5, i, lots, conf)

            elif signal == 0 and position is None:
                position = (-1, price - SPREAD_COST * 0.5, i, lots, conf)

            elif signal == 2 and position and position[0] == -1:
                ctx = self._make_exit_context(position, price, i, "signal_reverse", balance)
                if self.hitl.check_exit(ctx):
                    pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], price, position[3])
                    balance += pnl_usd
                    peak_balance = max(peak_balance, balance)
                    trades.append(Trade(position[2], i, position[0], position[1], price,
                                        pnl_pips, pnl_usd, i-position[2], position[4],
                                        "signal_reverse", position[3], balance))
                    position = (1, price + SPREAD_COST * 0.5, i, lots, conf)

            elif signal == 0 and position and position[0] == 1:
                ctx = self._make_exit_context(position, price, i, "signal_reverse", balance)
                if self.hitl.check_exit(ctx):
                    pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], price, position[3])
                    balance += pnl_usd
                    peak_balance = max(peak_balance, balance)
                    trades.append(Trade(position[2], i, position[0], position[1], price,
                                        pnl_pips, pnl_usd, i-position[2], position[4],
                                        "signal_reverse", position[3], balance))
                    position = (-1, price - SPREAD_COST * 0.5, i, lots, conf)

            equity.append(balance)

        # Close remaining
        if position:
            pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], prices[-1], position[3])
            balance += pnl_usd
            trades.append(Trade(position[2], len(prices)-1, position[0], position[1], prices[-1],
                                pnl_pips, pnl_usd, len(prices)-1-position[2], position[4],
                                "end_of_data", position[3], balance))
            equity.append(balance)

        if filtered_count > 0:
            logger.info(f"Confidence filter: {filtered_count} low-conf signals → hold")

        if self.hitl.enabled:
            logger.info(
                f"HITL stats: approved={self.hitl.stats['approved']} "
                f"vetoed={self.hitl.stats['vetoed']} "
                f"auto={self.hitl.stats['auto_approved']}"
            )

        return BacktestResult(
            trades=trades, equity_curve=equity,
            initial_balance=cfg.initial_balance, final_balance=balance,
        )
