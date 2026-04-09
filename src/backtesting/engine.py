"""Backtesting engine with HITL exit approval and confidence-based sizing.

Key differences from v1:
    - Confidence modulates position size (high conf = larger)
    - HITL gate intercepts exits on losing positions
    - Tracks confidence calibration metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from loguru import logger

from src.hitl.mt5_interface import HITLGate, SignalContext


@dataclass
class BacktestConfig:
    initial_balance: float = 10_000.0
    base_lot_size: float = 0.01
    max_lots: float = 0.10
    spread_pips: float = 2.0
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0
    pip_value: float = 0.10
    pip_usd_per_lot: float = 10.0  # per 0.01 lot
    max_position_time: int = 120
    human_exit_approval: bool = False


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


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_balance: float = 10_000.0
    final_balance: float = 10_000.0

    @property
    def total_trades(self): return len(self.trades)

    @property
    def win_rate(self):
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)
        return wins / max(1, self.total_trades)

    @property
    def profit_factor(self):
        gross_p = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        gross_l = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd <= 0))
        return gross_p / max(gross_l, 1e-8)

    @property
    def total_pnl(self): return sum(t.pnl_usd for t in self.trades)

    @property
    def max_drawdown(self):
        if not self.equity_curve: return 0.0
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        return float(((peak - eq) / peak).max())

    @property
    def sharpe_ratio(self):
        if len(self.trades) < 2: return 0.0
        rets = [t.pnl_usd for t in self.trades]
        m, s = np.mean(rets), np.std(rets)
        return 0.0 if s == 0 else m / s * np.sqrt(252 * 20)

    def summary(self) -> str:
        avg_w = np.mean([t.pnl_usd for t in self.trades if t.pnl_usd > 0]) if any(t.pnl_usd > 0 for t in self.trades) else 0
        avg_l = np.mean([t.pnl_usd for t in self.trades if t.pnl_usd <= 0]) if any(t.pnl_usd <= 0 for t in self.trades) else 0
        return (
            f"{'='*50}\nBACKTEST RESULTS\n{'='*50}\n"
            f"Total Trades:    {self.total_trades}\n"
            f"Win Rate:        {self.win_rate:.1%}\n"
            f"Profit Factor:   {self.profit_factor:.2f}\n"
            f"Total PnL:       ${self.total_pnl:.2f}\n"
            f"Avg Win:         ${avg_w:.2f}\n"
            f"Avg Loss:        ${avg_l:.2f}\n"
            f"Max Drawdown:    {self.max_drawdown:.1%}\n"
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}\n"
            f"Final Balance:   ${self.final_balance:.2f}\n"
            f"{'='*50}"
        )


class BacktestEngine:
    """Backtesting with confidence-based sizing and HITL approval."""

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
            prices: Close prices (n,)
            signals: Actions 0=sell, 1=hold, 2=buy (n,)
            confidences: Confidence scores [0,1] (n,) — modulates lot size
        """
        cfg = self.config
        if confidences is None:
            confidences = np.ones(len(prices))

        balance = cfg.initial_balance
        position = None  # (direction, entry_price, entry_idx, lots, confidence)
        trades = []
        equity = [balance]

        for i in range(len(prices)):
            signal = int(signals[i])
            price = float(prices[i])
            conf = float(confidences[i])

            # Force close on max hold
            if position and (i - position[2]) >= cfg.max_position_time:
                ctx = SignalContext(
                    action="CLOSE", confidence=position[4],
                    current_price=price, entry_price=position[1],
                    unrealized_pnl=self._calc_pnl(position, price),
                    hold_time_bars=i - position[2], exit_reason="max_time",
                )
                if self.hitl.check_exit(ctx):
                    pnl = self._close(position, price, i, "max_time")
                    trades.append(pnl)
                    balance += pnl.pnl_usd
                    position = None

            # Confidence-based lot sizing
            lots = min(cfg.base_lot_size * (1 + conf * 4), cfg.max_lots)  # 0.01–0.05 range
            lots = round(max(0.01, lots), 2)

            # Execute signals
            if signal == 2 and position is None:  # Buy
                entry = price + cfg.spread_pips * cfg.pip_value * 0.5
                position = (1, entry, i, lots, conf)

            elif signal == 0 and position is None:  # Sell short
                entry = price - cfg.spread_pips * cfg.pip_value * 0.5
                position = (-1, entry, i, lots, conf)

            elif signal == 2 and position and position[0] == -1:  # Close short → open long
                ctx = SignalContext(
                    action="CLOSE_SHORT", confidence=conf,
                    current_price=price, entry_price=position[1],
                    unrealized_pnl=self._calc_pnl(position, price),
                    hold_time_bars=i - position[2], exit_reason="signal_reverse",
                )
                if self.hitl.check_exit(ctx):
                    pnl = self._close(position, price, i, "signal_reverse")
                    trades.append(pnl)
                    balance += pnl.pnl_usd
                    entry = price + cfg.spread_pips * cfg.pip_value * 0.5
                    position = (1, entry, i, lots, conf)

            elif signal == 0 and position and position[0] == 1:  # Close long → open short
                ctx = SignalContext(
                    action="CLOSE_LONG", confidence=conf,
                    current_price=price, entry_price=position[1],
                    unrealized_pnl=self._calc_pnl(position, price),
                    hold_time_bars=i - position[2], exit_reason="signal_reverse",
                )
                if self.hitl.check_exit(ctx):
                    pnl = self._close(position, price, i, "signal_reverse")
                    trades.append(pnl)
                    balance += pnl.pnl_usd
                    entry = price - cfg.spread_pips * cfg.pip_value * 0.5
                    position = (-1, entry, i, lots, conf)

            equity.append(balance)

        # Close remaining
        if position:
            pnl = self._close(position, prices[-1], len(prices) - 1, "end_of_data")
            trades.append(pnl)
            balance += pnl.pnl_usd
            equity.append(balance)

        result = BacktestResult(trades=trades, equity_curve=equity,
                                initial_balance=cfg.initial_balance, final_balance=balance)
        logger.info(f"\n{result.summary()}")
        return result

    def _calc_pnl(self, position, price):
        d, ep, _, lots, _ = position
        return (price - ep) * d / self.config.pip_value * self.config.pip_usd_per_lot * (lots / 0.01)

    def _close(self, position, price, idx, reason):
        cfg = self.config
        d, ep, ei, lots, conf = position
        cost = cfg.spread_pips * cfg.pip_value * 0.5 + np.random.uniform(0, cfg.slippage_pips * cfg.pip_value)
        exit_p = price - cost * d
        pnl_pips = (exit_p - ep) * d / cfg.pip_value
        pnl_usd = pnl_pips * cfg.pip_usd_per_lot * (lots / 0.01) - cfg.commission_per_lot
        return Trade(ei, idx, d, ep, exit_p, pnl_pips, pnl_usd, idx - ei, conf, reason)
