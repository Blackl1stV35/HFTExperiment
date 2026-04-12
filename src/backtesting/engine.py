"""Backtesting engine v2 — PnL matched to RL training environment.

CRITICAL: The PnL calculation here MUST match the RL FrozenEncoderEnv.
Previous bug: backtest used $10/pip while RL used $1/pip → 10x mismatch.

For XAUUSD 0.01 lot:
    1 pip = $0.10 price move
    PnL per pip = $1.00 (for 0.01 lot)
    100 pip move = $100 PnL

Confidence filtering:
    --min-confidence flag skips signals below threshold (converts to hold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from src.hitl.mt5_interface import HITLGate, SignalContext


# Execution constants — MUST match RL FrozenEncoderEnv
PIP_VALUE = 0.10           # 1 pip = $0.10 price move
PIP_USD_PER_MICROLOT = 1.0 # $1 per pip per 0.01 lot
SPREAD_PIPS = 2.0
SPREAD_COST = SPREAD_PIPS * PIP_VALUE  # $0.20
COMMISSION = 0.70          # per 0.01 lot round-trip
SLIPPAGE_PIPS = 0.5


@dataclass
class BacktestConfig:
    initial_balance: float = 10_000.0
    base_lot_size: float = 0.01
    max_lots: float = 0.05
    max_position_time: int = 120
    human_exit_approval: bool = False
    min_confidence: float = 0.0  # filter: skip signals below this


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_usd: float
    hold_time: int
    confidence: float = 0.0
    exit_reason: str = ""
    lots: float = 0.01


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
        gross_p = sum(t.pnl_usd for t in self.winning_trades)
        gross_l = abs(sum(t.pnl_usd for t in self.losing_trades))
        return gross_p / max(gross_l, 1e-8)

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
            f"{'='*55}"
        )

    def profitable_trades_summary(self) -> str:
        """Summary of only profitable trades."""
        wins = self.winning_trades
        if not wins:
            return "No profitable trades."
        avg_pnl = np.mean([t.pnl_usd for t in wins])
        avg_conf = np.mean([t.confidence for t in wins])
        avg_hold = np.mean([t.hold_time for t in wins])
        return (
            f"  Profitable trades: {len(wins)}/{self.total_trades}\n"
            f"  Avg win PnL:       ${avg_pnl:.2f}\n"
            f"  Avg win confidence:{avg_conf:.3f}\n"
            f"  Avg win hold time: {avg_hold:.0f} bars"
        )


def _compute_pnl(direction: int, entry_price: float, exit_price: float, lots: float) -> tuple[float, float]:
    """Compute PnL in pips and USD — MATCHES RL FrozenEncoderEnv calculation.

    For XAUUSD 0.01 lot: $1 per pip.
    """
    pnl_pips = (exit_price - entry_price) * direction / PIP_VALUE
    pnl_usd = pnl_pips * PIP_USD_PER_MICROLOT * (lots / 0.01)
    pnl_usd -= COMMISSION * (lots / 0.01)  # commission scaled to lot
    return pnl_pips, pnl_usd


class BacktestEngine:
    """Backtesting with confidence-based sizing, filtering, and HITL."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.hitl = HITLGate(enabled=self.config.human_exit_approval)

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            prices: (n,) close prices
            signals: (n,) actions: 0=sell, 1=hold, 2=buy
            confidences: (n,) confidence [0,1] — modulates lot size and filtering
        """
        cfg = self.config
        if confidences is None:
            confidences = np.ones(len(prices), dtype=np.float32) * 0.5

        balance = cfg.initial_balance
        position = None  # (direction, entry_price, entry_idx, lots, confidence)
        trades = []
        equity = [balance]

        filtered_count = 0

        for i in range(len(prices)):
            signal = int(signals[i])
            price = float(prices[i])
            conf = float(confidences[i])

            # ── Confidence filter: convert low-confidence to hold ──
            if conf < cfg.min_confidence and signal != 1:
                signal = 1  # override to hold
                filtered_count += 1

            # ── Force close on max hold ──
            if position and (i - position[2]) >= cfg.max_position_time:
                exit_price = price - SLIPPAGE_PIPS * PIP_VALUE * position[0]
                pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], exit_price, position[3])

                ctx = SignalContext(
                    action="CLOSE", confidence=position[4],
                    current_price=price, entry_price=position[1],
                    unrealized_pnl=pnl_usd,
                    hold_time_bars=i - position[2],
                    exit_reason="max_hold_time",
                    position_size_lots=position[3],
                )
                if self.hitl.check_exit(ctx):
                    trades.append(Trade(position[2], i, position[0], position[1], exit_price,
                                        pnl_pips, pnl_usd, i - position[2], position[4], "max_time", position[3]))
                    balance += pnl_usd
                    position = None

            # ── Confidence-based lot sizing ──
            # Scale: conf=0.3 → 0.01 lots, conf=1.0 → max_lots
            if conf > cfg.min_confidence:
                conf_scale = min((conf - cfg.min_confidence) / (1.0 - cfg.min_confidence + 1e-8), 1.0)
                lots = cfg.base_lot_size + conf_scale * (cfg.max_lots - cfg.base_lot_size)
                lots = round(max(cfg.base_lot_size, min(lots, cfg.max_lots)), 2)
            else:
                lots = cfg.base_lot_size

            # ── Execute signals ──
            if signal == 2 and position is None:  # Buy
                entry = price + SPREAD_COST * 0.5
                position = (1, entry, i, lots, conf)

            elif signal == 0 and position is None:  # Sell short
                entry = price - SPREAD_COST * 0.5
                position = (-1, entry, i, lots, conf)

            elif signal == 2 and position and position[0] == -1:  # Close short → long
                exit_price = price + SLIPPAGE_PIPS * PIP_VALUE * 0.5
                pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], exit_price, position[3])

                ctx = SignalContext(
                    action="CLOSE_SHORT", confidence=conf,
                    current_price=price, entry_price=position[1],
                    unrealized_pnl=pnl_usd,
                    hold_time_bars=i - position[2],
                    exit_reason="signal_reverse",
                    position_size_lots=position[3],
                )
                if self.hitl.check_exit(ctx):
                    trades.append(Trade(position[2], i, position[0], position[1], exit_price,
                                        pnl_pips, pnl_usd, i - position[2], position[4], "signal_reverse", position[3]))
                    balance += pnl_usd
                    entry = price + SPREAD_COST * 0.5
                    position = (1, entry, i, lots, conf)

            elif signal == 0 and position and position[0] == 1:  # Close long → short
                exit_price = price - SLIPPAGE_PIPS * PIP_VALUE * 0.5
                pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], exit_price, position[3])

                ctx = SignalContext(
                    action="CLOSE_LONG", confidence=conf,
                    current_price=price, entry_price=position[1],
                    unrealized_pnl=pnl_usd,
                    hold_time_bars=i - position[2],
                    exit_reason="signal_reverse",
                    position_size_lots=position[3],
                )
                if self.hitl.check_exit(ctx):
                    trades.append(Trade(position[2], i, position[0], position[1], exit_price,
                                        pnl_pips, pnl_usd, i - position[2], position[4], "signal_reverse", position[3]))
                    balance += pnl_usd
                    entry = price - SPREAD_COST * 0.5
                    position = (-1, entry, i, lots, conf)

            equity.append(balance)

        # Close remaining
        if position:
            exit_price = prices[-1]
            pnl_pips, pnl_usd = _compute_pnl(position[0], position[1], exit_price, position[3])
            trades.append(Trade(position[2], len(prices)-1, position[0], position[1], exit_price,
                                pnl_pips, pnl_usd, len(prices)-1-position[2], position[4], "end_of_data", position[3]))
            balance += pnl_usd
            equity.append(balance)

        if filtered_count > 0:
            logger.info(f"Confidence filter: {filtered_count} low-conf signals converted to hold")

        result = BacktestResult(
            trades=trades, equity_curve=equity,
            initial_balance=cfg.initial_balance, final_balance=balance,
        )
        logger.info(f"\n{result.summary()}")
        return result
