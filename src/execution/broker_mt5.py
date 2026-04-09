"""MetaTrader 5 broker interface for order execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class OrderResult:
    """Result of an order execution."""

    success: bool
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    comment: str = ""
    latency_ms: float = 0.0


class MT5Broker:
    """MetaTrader 5 broker connection and order execution."""

    def __init__(
        self,
        symbol: str = "XAUUSD",
        magic_number: int = 20240101,
        max_slippage_points: int = 30,
    ):
        self.symbol = symbol
        self.magic_number = magic_number
        self.max_slippage = max_slippage_points
        self._mt5 = None
        self._connected = False

    def connect(self, login: str, password: str, server: str, path: Optional[str] = None) -> bool:
        """Initialize MT5 connection."""
        try:
            import MetaTrader5 as mt5

            self._mt5 = mt5
            init_kwargs = {}
            if path:
                init_kwargs["path"] = path

            if not mt5.initialize(**init_kwargs):
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False

            if not mt5.login(int(login), password=password, server=server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False

            # Verify symbol is available
            info = mt5.symbol_info(self.symbol)
            if info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return False

            if not info.visible:
                mt5.symbol_select(self.symbol, True)

            self._connected = True
            logger.info(f"Connected to MT5: {server}")
            return True

        except ImportError:
            logger.error("MetaTrader5 not installed")
            return False

    def get_tick(self) -> Optional[dict]:
        """Get current bid/ask tick."""
        if not self._connected:
            return None

        tick = self._mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "time": tick.time,
            "spread": round((tick.ask - tick.bid) / 0.01),
        }

    def buy(self, volume: float, comment: str = "") -> OrderResult:
        """Execute a market buy order."""
        return self._send_order(
            action_type="BUY",
            volume=volume,
            comment=comment,
        )

    def sell(self, volume: float, comment: str = "") -> OrderResult:
        """Execute a market sell order."""
        return self._send_order(
            action_type="SELL",
            volume=volume,
            comment=comment,
        )

    def close_position(self, ticket: int) -> OrderResult:
        """Close an open position by ticket."""
        if not self._connected:
            return OrderResult(success=False, comment="Not connected")

        mt5 = self._mt5
        position = None

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(success=False, comment=f"Position {ticket} not found")

        position = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol)
        close_price = price.bid if close_type == mt5.ORDER_TYPE_SELL else price.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": self.max_slippage,
            "magic": self.magic_number,
            "comment": "close",
        }

        start = time.perf_counter()
        result = mt5.order_send(request)
        latency = (time.perf_counter() - start) * 1000

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = result.comment if result else "Unknown error"
            logger.error(f"Close failed: {error}")
            return OrderResult(success=False, comment=error, latency_ms=latency)

        return OrderResult(
            success=True,
            ticket=result.order,
            price=result.price,
            volume=position.volume,
            latency_ms=latency,
        )

    def _send_order(self, action_type: str, volume: float, comment: str) -> OrderResult:
        """Send a market order to MT5."""
        if not self._connected:
            return OrderResult(success=False, comment="Not connected")

        mt5 = self._mt5
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return OrderResult(success=False, comment="No tick data")

        if action_type == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": self.max_slippage,
            "magic": self.magic_number,
            "comment": comment or f"NN_{action_type}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        start = time.perf_counter()
        result = mt5.order_send(request)
        latency = (time.perf_counter() - start) * 1000

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = result.comment if result else "Unknown error"
            logger.error(f"Order failed: {error}")
            return OrderResult(success=False, comment=error, latency_ms=latency)

        logger.info(
            f"Order executed: {action_type} {volume} @ {result.price} "
            f"(ticket={result.order}, latency={latency:.1f}ms)"
        )
        return OrderResult(
            success=True,
            ticket=result.order,
            price=result.price,
            volume=volume,
            latency_ms=latency,
        )

    def get_open_positions(self) -> list[dict]:
        """Get all open positions for this EA."""
        if not self._connected:
            return []

        positions = self._mt5.positions_get(symbol=self.symbol)
        if not positions:
            return []

        return [
            {
                "ticket": p.ticket,
                "type": "buy" if p.type == 0 else "sell",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "profit": p.profit,
                "magic": p.magic,
            }
            for p in positions
            if p.magic == self.magic_number
        ]

    def get_account_info(self) -> dict:
        """Get account balance and equity."""
        if not self._connected:
            return {}

        info = self._mt5.account_info()
        if info is None:
            return {}

        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "leverage": info.leverage,
        }

    def disconnect(self) -> None:
        """Close MT5 connection."""
        if self._mt5:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")
