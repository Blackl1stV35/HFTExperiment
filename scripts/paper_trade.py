#!/usr/bin/env python3
"""Paper/live trading with the dual-branch model + HITL.

Usage:
    python scripts/paper_trade.py --config configs/deployment/production.yaml --synthetic
    python scripts/paper_trade.py --config configs/deployment/production.yaml
"""

from __future__ import annotations

import argparse
import signal as sig
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from loguru import logger

from src.data.preprocessing import WindowMinMaxScaler
from src.hitl.mt5_interface import HITLGate, SignalContext, RiskDisplay
from src.inference.onnx_engine import ONNXInferenceEngine
from src.risk.circuit_breaker import CircuitBreaker, PositionSizer
from src.monitoring.alerts import TelegramAlerter
from src.utils.config import load_env, setup_logger


class TradingLoop:
    """Main trading loop with dual-branch model + HITL approval."""

    def __init__(self, config_path: str, synthetic: bool = False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.synthetic = synthetic
        self.running = False

        self.engine = None
        self.broker = None
        self.circuit_breaker = None
        self.hitl = None
        self.scaler = WindowMinMaxScaler(window_size=120)
        self.price_buffer = []
        self.feature_buffer = []

    def initialize(self) -> bool:
        load_env()
        risk_cfg = self.config.get("risk", {})
        hitl_cfg = self.config.get("hitl", {})

        # Inference engine
        model_path = self.config.get("inference", {}).get("model_path", "exports/best_model.onnx")
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            return False

        self.engine = ONNXInferenceEngine(model_path, device="cpu", n_threads=4)

        # Risk
        self.circuit_breaker = CircuitBreaker(
            max_daily_drawdown_pct=risk_cfg.get("max_daily_drawdown_pct", 2.0),
            max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 5),
            latency_kill_ms=risk_cfg.get("latency_kill_ms", 50.0),
        )

        # HITL gate
        self.hitl = HITLGate(
            enabled=hitl_cfg.get("enabled", True) and not self.synthetic,
            auto_approve_confidence=hitl_cfg.get("auto_approve_confidence", 0.7),
            auto_approve_max_lots=hitl_cfg.get("auto_approve_max_lots", 0.03),
        )

        # Broker
        if self.synthetic:
            self.broker = SyntheticBroker()
        else:
            from src.execution.broker_mt5 import MT5Broker
            from src.utils.config import BrokerConfig
            bc = BrokerConfig.from_env()
            self.broker = MT5Broker()
            if not self.broker.connect(bc.login, bc.password, bc.server, bc.path):
                return False

        # Alerter
        self.alerter = TelegramAlerter()
        if not self.synthetic:
            self.alerter.alert_startup()

        logger.info("All components initialized")
        return True

    def run(self) -> None:
        self.running = True
        sig.signal(sig.SIGINT, lambda s, f: setattr(self, "running", False))
        sig.signal(sig.SIGTERM, lambda s, f: setattr(self, "running", False))

        tick_count = 0
        position = None  # (direction, entry_price, lots)

        while self.running:
            try:
                tick = self._get_tick()
                if tick is None:
                    time.sleep(0.1)
                    continue

                tick_count += 1
                self.price_buffer.append(tick["bid"])
                self.feature_buffer.append([
                    tick["bid"], tick["bid"] + 0.5, tick["bid"] - 0.5,
                    tick["bid"], 100, tick.get("spread", 20),
                ])

                if len(self.price_buffer) > 300:
                    self.price_buffer = self.price_buffer[-300:]
                    self.feature_buffer = self.feature_buffer[-300:]

                seq_len = 120
                if len(self.feature_buffer) < seq_len:
                    continue

                # Prepare input
                features = np.array(self.feature_buffer[-seq_len:], dtype=np.float32)
                scaled = self.scaler.transform(features).astype(np.float32)
                input_seq = scaled.reshape(1, seq_len, -1)

                # Inference — dual output: signal + confidence
                output, latency = self.engine.predict_timed(x=input_seq)

                # Parse dual output: first 3 = logits, last 1 = confidence
                if output.shape[-1] >= 4:
                    logits = output[0, :3]
                    confidence = float(1 / (1 + np.exp(-output[0, 3])))  # sigmoid
                else:
                    logits = output[0]
                    confidence = float(np.max(np.exp(logits) / np.exp(logits).sum()))

                probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
                action = int(probs.argmax())

                # Circuit breaker
                can_trade, reason = self.circuit_breaker.check_can_trade(latency)
                if not can_trade:
                    if tick_count % 100 == 0:
                        logger.warning(f"Blocked: {reason}")
                    continue

                # Display signal
                if tick_count % 50 == 0:
                    display = RiskDisplay.format_signal(action, confidence, tick["bid"])
                    logger.info(f"Tick {tick_count} | {display} | Latency: {latency:.1f}ms")

                # Execute with HITL
                if action != 1 and position is None:  # Entry
                    ctx = SignalContext(
                        action="BUY" if action == 2 else "SELL",
                        confidence=confidence,
                        current_price=tick["bid"],
                        position_size_lots=min(0.01 * (1 + confidence * 4), 0.05),
                    )
                    if self.hitl.check_entry(ctx):
                        if action == 2:
                            result = self.broker.buy(ctx.position_size_lots)
                        else:
                            result = self.broker.sell(ctx.position_size_lots)
                        if hasattr(result, "success") and result.success:
                            position = (1 if action == 2 else -1, tick["bid"], ctx.position_size_lots)

                elif action != 1 and position and action != (1 + position[0]):  # Exit signal
                    ctx = SignalContext(
                        action="CLOSE",
                        confidence=confidence,
                        current_price=tick["bid"],
                        entry_price=position[1],
                        unrealized_pnl=(tick["bid"] - position[1]) * position[0],
                        exit_reason="signal_reverse",
                    )
                    if self.hitl.check_exit(ctx):
                        positions = self.broker.get_open_positions()
                        for p in positions:
                            self.broker.close_position(p["ticket"])
                        position = None

            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                time.sleep(1)

        logger.info(f"Shutdown. HITL stats: {self.hitl.stats}")
        if not self.synthetic:
            self.alerter.alert_shutdown()

    def _get_tick(self):
        if self.synthetic:
            if not self.price_buffer:
                price = 2000.0
            else:
                price = self.price_buffer[-1] + np.random.normal(0, 0.1)
            time.sleep(0.01)
            return {"bid": price, "ask": price + 0.02, "spread": 20}
        return self.broker.get_tick()


class SyntheticBroker:
    def __init__(self):
        self._positions = {}
        self._ticket = 1000

    def buy(self, vol, comment=""):
        from src.execution.broker_mt5 import OrderResult
        self._ticket += 1
        return OrderResult(success=True, ticket=self._ticket, price=2000, volume=vol, latency_ms=0.1)

    def sell(self, vol, comment=""):
        from src.execution.broker_mt5 import OrderResult
        self._ticket += 1
        return OrderResult(success=True, ticket=self._ticket, price=2000, volume=vol, latency_ms=0.1)

    def close_position(self, ticket):
        from src.execution.broker_mt5 import OrderResult
        return OrderResult(success=True, ticket=ticket, price=2000, latency_ms=0.1)

    def get_open_positions(self):
        return []

    def get_tick(self):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deployment/production.yaml")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    setup_logger()
    loop = TradingLoop(args.config, synthetic=args.synthetic)
    if loop.initialize():
        loop.run()


if __name__ == "__main__":
    main()
