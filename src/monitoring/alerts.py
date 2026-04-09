"""Monitoring: Prometheus metrics + Telegram alerts + Grafana-ready exports."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from loguru import logger


class MetricsCollector:
    """Prometheus metrics for Grafana Cloud monitoring."""

    def __init__(self, port: int = 9090):
        self.port = port
        self._started = False

        try:
            from prometheus_client import (
                Counter, Gauge, Histogram, Summary, start_http_server,
            )

            # Trading metrics
            self.trades_total = Counter(
                "trading_trades_total", "Total trades executed", ["direction", "result"]
            )
            self.pnl_total = Gauge("trading_pnl_total", "Total session PnL in USD")
            self.pnl_daily = Gauge("trading_pnl_daily", "Daily PnL in USD")
            self.balance = Gauge("trading_account_balance", "Account balance")
            self.position = Gauge("trading_position", "Current position direction")
            self.drawdown = Gauge("trading_max_drawdown", "Maximum drawdown percentage")

            # Model metrics
            self.inference_latency = Histogram(
                "trading_inference_latency_ms",
                "Model inference latency in milliseconds",
                buckets=[0.5, 1, 2, 3, 5, 10, 20, 50],
            )
            self.model_uncertainty = Gauge("trading_model_uncertainty", "Model prediction uncertainty")
            self.regime = Gauge("trading_regime", "Current market regime", ["regime_name"])

            # System metrics
            self.ticks_processed = Counter("trading_ticks_total", "Total ticks processed")
            self.orders_sent = Counter("trading_orders_sent", "Orders sent to broker")
            self.orders_rejected = Counter("trading_orders_rejected", "Orders rejected by broker")
            self.circuit_breaker_halts = Counter("trading_circuit_breaker_halts", "Circuit breaker activations")
            self.connection_errors = Counter("trading_connection_errors", "Broker connection errors")

            self._prom_available = True
        except ImportError:
            logger.warning("prometheus_client not installed — metrics disabled")
            self._prom_available = False

    def start(self) -> None:
        """Start Prometheus HTTP server."""
        if self._prom_available and not self._started:
            from prometheus_client import start_http_server

            start_http_server(self.port)
            self._started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")

    def record_trade(self, direction: str, result: str, pnl: float) -> None:
        """Record a completed trade."""
        if self._prom_available:
            self.trades_total.labels(direction=direction, result=result).inc()
            self.pnl_total.set(self.pnl_total._value.get() + pnl if hasattr(self.pnl_total, '_value') else pnl)

    def record_inference(self, latency_ms: float, uncertainty: float) -> None:
        """Record model inference metrics."""
        if self._prom_available:
            self.inference_latency.observe(latency_ms)
            self.model_uncertainty.set(uncertainty)

    def record_tick(self) -> None:
        if self._prom_available:
            self.ticks_processed.inc()

    def update_balance(self, balance: float) -> None:
        if self._prom_available:
            self.balance.set(balance)

    def record_regime(self, regime_name: str) -> None:
        if self._prom_available:
            # Reset all regime gauges
            for r in ["trending_up", "trending_down", "ranging", "volatile"]:
                self.regime.labels(regime_name=r).set(0)
            self.regime.labels(regime_name=regime_name.lower()).set(1)


class TelegramAlerter:
    """Send alerts via Telegram bot for critical events."""

    ALERT_TYPES = {
        "trade": "📊",
        "risk": "🚨",
        "error": "❌",
        "info": "ℹ️",
        "startup": "🟢",
        "shutdown": "🔴",
        "profit": "💰",
        "loss": "📉",
    }

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self._enabled = bool(self.bot_token and self.chat_id)

        if not self._enabled:
            logger.info("Telegram alerts disabled (no token/chat_id)")

    def send(self, message: str, alert_type: str = "info") -> bool:
        """Send a Telegram message."""
        if not self._enabled:
            return False

        emoji = self.ALERT_TYPES.get(alert_type, "📋")
        full_msg = f"{emoji} *XAUUSD Trading Bot*\n\n{message}"

        try:
            import requests

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            resp = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": full_msg,
                    "parse_mode": "Markdown",
                },
                timeout=5,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def alert_trade(self, direction: str, price: float, volume: float, pnl: Optional[float] = None) -> None:
        """Alert on trade execution."""
        msg = f"Trade: {direction.upper()} {volume} lots @ {price:.2f}"
        if pnl is not None:
            msg += f"\nPnL: ${pnl:.2f}"
        alert_type = "profit" if pnl and pnl > 0 else "loss" if pnl and pnl < 0 else "trade"
        self.send(msg, alert_type)

    def alert_risk(self, reason: str) -> None:
        """Alert on risk event (circuit breaker, uncertainty, etc)."""
        self.send(f"⚠️ Risk Alert: {reason}", "risk")

    def alert_error(self, error: str) -> None:
        """Alert on system error."""
        self.send(f"Error: {error}", "error")

    def alert_daily_summary(self, stats: dict) -> None:
        """Send end-of-day summary."""
        msg = (
            f"📊 *Daily Summary*\n"
            f"PnL: ${stats.get('pnl', 0):.2f}\n"
            f"Trades: {stats.get('trades', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1%}\n"
            f"Max DD: {stats.get('max_drawdown', 0):.1%}\n"
            f"Balance: ${stats.get('balance', 0):.2f}"
        )
        self.send(msg, "info")

    def alert_startup(self) -> None:
        self.send("Trading bot started ✅", "startup")

    def alert_shutdown(self, reason: str = "normal") -> None:
        self.send(f"Trading bot stopped. Reason: {reason}", "shutdown")
