#!/usr/bin/env python3
"""Paper/live trading — Phase 8b deployment.

Inference stack:
    1. MT5 M1 bars → 10-dim feature builder (matches precompute_features.py)
    2. Frozen DualBranchModel (Run 10 ep61, dual_branch_last.pt)
       → 3-class probs + confidence
    3. Regime deploy gate: Bear+HIGH → skip bar
    4. ConfidenceSACAgent (Phase 8b evolve2, rl_agent_evolve2.pt)
       → action[0]=position_size, action[1]=exit_logit

Usage:
    python scripts/paper_trade.py --config configs/deployment/production.yaml --synthetic
    python scripts/paper_trade.py --config configs/deployment/production.yaml
"""

from __future__ import annotations

import argparse
import csv
import signal as sig
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from loguru import logger

from src.hitl.mt5_interface import HITLGate, SignalContext, DrawdownContext
from src.risk.circuit_breaker import CircuitBreaker, PositionSizer
from src.monitoring.alerts import TelegramAlerter
from src.utils.config import load_env, setup_logger


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_supervised(checkpoint_path: str, cfg_path: str):
    """Load frozen DualBranchModel from .pt checkpoint.

    Uses DualBranchModel.from_config() so classifier_dims and all arch
    params are read from the yaml correctly (avoids __init__ default mismatch).
    Falls back to strict=False with missing/unexpected key logging if there
    is any remaining mismatch (e.g. local vs Colab yaml drift).
    """
    from omegaconf import OmegaConf
    from src.encoder.fusion import DualBranchModel

    cfg   = OmegaConf.load(cfg_path)
    model = DualBranchModel.from_config(cfg)
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False
    )
    if missing:
        logger.warning(f"  Missing keys in checkpoint: {missing}")
    if unexpected:
        logger.warning(f"  Unexpected keys in checkpoint: {unexpected}")
    if not missing and not unexpected:
        logger.info("  State dict loaded cleanly (strict match)")
    model.eval()
    logger.info(f"Supervised model loaded: {checkpoint_path}")
    return model


def _load_rl_agent(checkpoint_path: str):
    """Load ConfidenceSACAgent — infers obs_dim from checkpoint first-layer shape.

    Supports both 16D (Phase 8b) and 18D (Phase 9) checkpoints automatically.
    """
    from src.meta_policy.rl_agent import ConfidenceSACAgent
    import torch

    # Peek at checkpoint to infer obs_dim from actor first-layer weight shape
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    obs_dim = 16  # safe default for Phase 8b
    for sd_key in ("actor_state_dict", "agent_state_dict", "state_dict"):
        sd = ckpt.get(sd_key, {})
        for k, v in sd.items():
            if hasattr(v, "shape") and v.ndim == 2:
                # First weight matrix in actor: shape = (hidden, obs_dim)
                obs_dim = int(v.shape[1])
                break
        if sd:
            break

    logger.info(f"RL checkpoint obs_dim inferred: {obs_dim}")
    agent = ConfidenceSACAgent(obs_dim=obs_dim, hidden_dims=[512, 512], device="cpu")
    agent.load(checkpoint_path)
    logger.info(f"RL agent loaded: {checkpoint_path}  (obs_dim={obs_dim})")
    return agent, obs_dim


# ── feature engineering ───────────────────────────────────────────────────────

def build_features(bars: list[dict]) -> np.ndarray | None:
    """Convert 240 raw M1 bars → (240, 10) normalised feature array.

    Features match precompute_features.py exactly:
        0  open_sc       RobustScaled open price
        1  high_sc       RobustScaled high
        2  low_sc        RobustScaled low
        3  close_sc      RobustScaled close
        4  vol_sc        RobustScaled tick_volume
        5  spread_sc     RobustScaled spread (points)
        6  bar_return_bps  log-return in basis points
        7  wick_asymmetry  (high-close - close-low) / (high-low)
        8  vol_zscore    tanh z-score of tick_volume over 20-bar window
        9  spread_pressure  spread_sc * vol_zscore
    """
    if len(bars) < 240:
        return None
    b = bars[-240:]

    opens   = np.array([x["open"]        for x in b], dtype=np.float64)
    highs   = np.array([x["high"]        for x in b], dtype=np.float64)
    lows    = np.array([x["low"]         for x in b], dtype=np.float64)
    closes  = np.array([x["close"]       for x in b], dtype=np.float64)
    vols    = np.array([x["tick_volume"] for x in b], dtype=np.float64)
    spreads = np.array([x["spread"]      for x in b], dtype=np.float64)

    def robust_scale(arr: np.ndarray) -> np.ndarray:
        med = np.median(arr)
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        return np.clip((arr - med) / (iqr + 1e-8), -3.0, 3.0).astype(np.float32)

    # bar_return_bps
    log_ret = np.concatenate([[0.0], np.log(closes[1:] / (closes[:-1] + 1e-8))]) * 10_000

    # wick_asymmetry
    hl = highs - lows + 1e-8
    wick_asym = ((highs - closes) - (closes - lows)) / hl

    # vol_zscore: tanh z over 20-bar rolling window
    vol_zsc = np.zeros(240, dtype=np.float32)
    for i in range(240):
        lo = max(0, i - 19)
        w  = vols[lo:i + 1]
        vol_zsc[i] = float(np.tanh((vols[i] - w.mean()) / (w.std() + 1e-8)))

    spread_sc_raw = robust_scale(spreads)
    spread_press  = spread_sc_raw * vol_zsc

    return np.stack([
        robust_scale(opens),
        robust_scale(highs),
        robust_scale(lows),
        robust_scale(closes),
        robust_scale(vols),
        spread_sc_raw,
        log_ret.astype(np.float32),
        wick_asym.astype(np.float32),
        vol_zsc,
        spread_press,
    ], axis=1).astype(np.float32)   # (240, 10)


# ── regime detection ──────────────────────────────────────────────────────────

def get_regime(bars: list[dict], lookback: int = 60) -> tuple[float, float]:
    """Estimate (gmm2, vol_enc) from recent bars.

    gmm2:    0.0=Bear (20-bar return < 0), 1.0=Bull
    vol_enc: 1.0=HIGH (ATR > 1.4× baseline), 0.0=LOW/NORMAL

    Returns (gmm2, vol_enc).
    """
    if len(bars) < lookback + 5:
        return 1.0, 0.0   # default: Bull, LOW — safe to trade

    closes = np.array([b["close"] for b in bars[-25:]])
    ret20  = (closes[-1] - closes[-20]) / (closes[-20] + 1e-8)
    gmm2   = 1.0 if ret20 >= 0 else 0.0

    recent = bars[-15:]
    base   = bars[-lookback:-15]
    atr_recent   = np.mean([b["high"] - b["low"] for b in recent])
    atr_baseline = np.mean([b["high"] - b["low"] for b in base]) + 1e-8
    # Phase 4 exploration: d_ATR directional gate (P1 Ma et al 2021, KS=0.149 STRONG)
    # Only block Bear+HIGH when ATR is ACTIVELY RISING (46% of Bear+HIGH bars
    # are falling-ATR recovery; blocking them unnecessarily reduces trade opportunities).
    atr_rising = atr_recent > atr_baseline   # volatility expanding vs contracting
    vol_enc = 1.0 if (atr_recent > atr_baseline * 1.4 and atr_rising) else 0.0

    return gmm2, vol_enc


# ── MTF return (ret_1h, ret_15m) ─────────────────────────────────────────────

def mtf_return_zscored(closes: np.ndarray, window: int) -> float:
    """Z-scored cumulative log-return over `window` bars, tanh-bounded."""
    if len(closes) < window + 10:
        return 0.0
    log_rets = np.log(closes[1:] / (closes[:-1] + 1e-8))
    cumret   = np.convolve(log_rets, np.ones(window), mode="valid")
    if len(cumret) < 2:
        return 0.0
    return float(np.tanh((cumret[-1] - cumret.mean()) / (cumret.std() + 1e-8)))


# ── supervised inference ──────────────────────────────────────────────────────

@torch.no_grad()
def run_supervised(model, features: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run DualBranchModel on (240, 10) feature array.

    Returns:
        probs      (3,) softmax probabilities [sell, hold, buy]
        confidence float scalar
        latency_ms float
    """
    x = torch.FloatTensor(features).unsqueeze(0)  # (1, 240, 10)
    t0 = time.perf_counter()
    logits, conf_logit = model(x)
    latency_ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    confidence = float(torch.sigmoid(conf_logit).squeeze())
    return probs, confidence, latency_ms


# ── RL obs builder ────────────────────────────────────────────────────────────

def build_rl_obs(
    probs: np.ndarray,
    confidence: float,
    position_dir: float,
    unrealized_pnl: float,
    hold_frac: float,
    atr_norm: float,
    trend_norm: float,
    session_phase: float,
    regime_quality: float,
    gs_quartile: float,
    cu_au_regime: float,
    ret_1h: float,
    ret_15m: float,
    session_phase_npz: float = 0.5,
    rq_regime: float = 0.5,
) -> np.ndarray:
    """Build RL observation vector — dimensionality matches the loaded checkpoint.

    Phase 8b / RL Phase 9 (16D):
        obs[0-3]   probs + confidence
        obs[4-6]   position_dir, unrealized_pnl, hold_frac
        obs[7-9]   atr_norm, trend_norm, session_phase
        obs[10-12] regime_quality, gs_quartile, cu_au_regime
        obs[13-14] ret_1h, ret_15m
        obs[15]    VIO (zero — bimodal instability)

    RL Phase 9+ (18D) adds:
        obs[16]    session_phase_npz
        obs[17]    rq_regime

    The obs_dim is inferred from the loaded RL checkpoint actor first-layer shape.
    """
    base = np.array([
        probs[0], probs[1], probs[2], confidence,           # 0-3
        position_dir, unrealized_pnl, hold_frac,            # 4-6
        atr_norm, trend_norm, session_phase,                 # 7-9
        regime_quality, gs_quartile, cu_au_regime,           # 10-12
        ret_1h, ret_15m,                                     # 13-14
        0.0,                                                 # 15 VIO disabled
    ], dtype=np.float32)
    # Append Phase 9 features only if checkpoint expects 18D
    # (checked at load time by _rl_obs_dim set in _load_rl_agent)
    ext = np.array([session_phase_npz, rq_regime], dtype=np.float32)
    return base  # extended to 18D in run() when self._rl_obs_dim == 18


# ── RL action ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_rl(agent, obs: np.ndarray) -> tuple[float, float, float]:
    """Run RL actor deterministically.

    Returns:
        size_signal  float in [-1, +1] (actor[0])
        exit_signal  float in [-1, +1] (actor[1])
        latency_ms   float
    """
    x = torch.FloatTensor(obs).unsqueeze(0)
    t0 = time.perf_counter()
    action = agent.actor(x).squeeze(0).numpy()
    latency_ms = (time.perf_counter() - t0) * 1000
    return float(action[0]), float(action[1]), latency_ms


# ── obs helper features ───────────────────────────────────────────────────────

def _atr_norm(bars: list[dict], window: int = 14) -> float:
    if len(bars) < window + 1:
        return 0.01
    trs = [b["high"] - b["low"] for b in bars[-window:]]
    close_avg = np.mean([b["close"] for b in bars[-window:]])
    return float(np.mean(trs) / (close_avg + 1e-8))


def _trend_norm(bars: list[dict], window: int = 20) -> float:
    if len(bars) < window + 1:
        return 0.0
    closes = np.array([b["close"] for b in bars[-window:]])
    ret = (closes[-1] - closes[0]) / (abs(closes[0]) + 1e-8)
    return float(np.tanh(ret * 50))


def _session_phase(bars: list[dict]) -> float:
    """0=Asian 0.5=London 1.0=NY based on UTC hour of latest bar."""
    if not bars:
        return 0.5
    ts = bars[-1].get("time", time.time())
    hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
    if 8 <= hour < 13:
        return 0.5   # London
    elif 13 <= hour < 22:
        return 1.0   # NY
    return 0.0       # Asian


# ── synthetic broker ──────────────────────────────────────────────────────────

class SyntheticBroker:
    def __init__(self):
        self._ticket = 1000
        self._price  = 3300.0

    def get_m1_bars(self, n: int) -> list[dict]:
        bars = []
        p = self._price
        for _ in range(n):
            o = p
            h = p + abs(np.random.normal(0, 0.5))
            l = p - abs(np.random.normal(0, 0.5))
            c = p + np.random.normal(0, 0.3)
            bars.append({"open": o, "high": h, "low": l, "close": c,
                         "tick_volume": max(1, int(np.random.normal(50, 20))),
                         "spread": 6, "time": time.time()})
            p = c
        self._price = p
        return bars

    def buy(self, vol, comment=""):
        from src.execution.broker_mt5 import OrderResult
        self._ticket += 1
        return OrderResult(success=True, ticket=self._ticket,
                           price=self._price, volume=vol, latency_ms=0.1)

    def sell(self, vol, comment=""):
        from src.execution.broker_mt5 import OrderResult
        self._ticket += 1
        return OrderResult(success=True, ticket=self._ticket,
                           price=self._price, volume=vol, latency_ms=0.1)

    def close_position(self, ticket, comment=""):
        from src.execution.broker_mt5 import OrderResult
        return OrderResult(success=True, ticket=ticket,
                           price=self._price, latency_ms=0.1)

    def get_open_positions(self):
        return []

    def get_account_info(self):
        return {"balance": 100_000.0, "equity": 100_000.0}


# ── MT5 bar fetcher ───────────────────────────────────────────────────────────

class MT5BarFetcher:
    """Wrapper around MT5 copy_rates_from_pos for M1 bar retrieval."""

    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self._mt5 = None

    def connect(self, login: str, password: str, server: str,
                path: str | None = None) -> bool:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            kwargs = {"path": path} if path else {}
            if not mt5.initialize(**kwargs):
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
            if not mt5.login(int(login), password=password, server=server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Symbol {self.symbol} not available")
                return False
            logger.info(f"MT5 connected to {server}")
            return True
        except ImportError:
            logger.error("MetaTrader5 not installed — run: pip install MetaTrader5")
            return False

    def get_m1_bars(self, n: int = 245) -> list[dict]:
        if self._mt5 is None:
            return []
        rates = self._mt5.copy_rates_from_pos(self.symbol, self._mt5.TIMEFRAME_M1, 0, n)
        if rates is None or len(rates) == 0:
            return []
        return [
            {
                "open":        float(r["open"]),
                "high":        float(r["high"]),
                "low":         float(r["low"]),
                "close":       float(r["close"]),
                "tick_volume": int(r["tick_volume"]),
                "spread":      int(r["spread"]),
                "time":        int(r["time"]),
            }
            for r in rates
        ]


# ── main trading loop ─────────────────────────────────────────────────────────

class TradingLoop:
    """Phase 8b paper trading loop.

    Decision pipeline each bar:
        M1 bars → feature build (240-bar) → supervised inference
        → regime gate → RL obs → RL action
        → HITL → MT5 order
    """

    def __init__(self, config_path: str, synthetic: bool = False):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.synthetic = synthetic
        self.running   = False

        self.sup_model  = None
        self.rl_agent   = None
        self.broker     = None
        self.bar_fetcher = None
        self.circuit_breaker = None
        self.hitl       = None
        self.alerter    = None

        # Position state
        self.position: dict | None = None  # {dir, entry_price, lots, ticket, hold_bars}
        self.hold_bars  = 0
        self.max_hold   = self.cfg.get("risk", {}).get("max_hold_bars", 80)
        self._paused     = False   # set True by /stop command
        self._force_close = False  # set True by /close command

        # Paper trade log
        log_path = Path(self.cfg.get("logging", {}).get("trade_log",
                                    "outputs/paper_trade_log.csv"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_path, "a", newline="")
        self._log = csv.writer(self._log_file)
        if log_path.stat().st_size == 0:
            self._log.writerow([
                "timestamp", "bar_time", "action", "direction",
                "entry_price", "close_price", "lots", "pnl_usd",
                "confidence", "rl_size", "rl_exit",
                "regime_gmm2", "regime_vol", "gated",
                "sell_p", "hold_p", "buy_p",
            ])
            self._log_file.flush()

    # ── init ──────────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        load_env()
        inf_cfg  = self.cfg.get("inference", {})
        risk_cfg = self.cfg.get("risk", {})
        hitl_cfg = self.cfg.get("hitl", {})
        mon_cfg  = self.cfg.get("monitoring", {})

        # Supervised model
        sup_ckpt  = inf_cfg.get("supervised_checkpoint", "models/dual_branch_last.pt")
        model_cfg = inf_cfg.get("model_config", "configs/model/dual_branch.yaml")
        if not Path(sup_ckpt).exists():
            logger.error(f"Supervised checkpoint not found: {sup_ckpt}")
            return False
        self.sup_model = _load_supervised(sup_ckpt, model_cfg)

        # RL agent
        rl_ckpt = inf_cfg.get("rl_checkpoint", "models/rl_agent_evolve2.pt")
        if not Path(rl_ckpt).exists():
            logger.error(f"RL checkpoint not found: {rl_ckpt}")
            return False
        self.rl_agent, self._rl_obs_dim = _load_rl_agent(rl_ckpt)

        # Circuit breaker
        acct_info = {"balance": 100_000.0}  # updated each loop from broker
        self.circuit_breaker = CircuitBreaker(
            max_daily_drawdown_pct=risk_cfg.get("max_daily_drawdown_pct", 2.0),
            max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 5),
            latency_kill_ms=risk_cfg.get("latency_kill_ms", 50.0),
            account_balance=acct_info["balance"],
        )

        # HITL — use Telegram approval when not in synthetic mode
        from src.hitl.mt5_interface import TelegramHITL
        tg_hitl = None
        if not self.synthetic and hitl_cfg.get("telegram_approval", True):
            tg_hitl = TelegramHITL(
                timeout_s=hitl_cfg.get("telegram_timeout_s", 120),
                auto_approve_on_timeout=hitl_cfg.get("auto_approve_on_timeout", True),
            )
            if tg_hitl._enabled:
                logger.info(f"TelegramHITL enabled — timeout={tg_hitl.timeout_s}s")
            else:
                logger.warning("TelegramHITL: env vars not set — falling back to console")
                tg_hitl = None

        self.hitl = HITLGate(
            enabled=hitl_cfg.get("enabled", True) and not self.synthetic,
            auto_approve_confidence=hitl_cfg.get("auto_approve_confidence", 0.72),
            auto_approve_max_lots=hitl_cfg.get("auto_approve_max_lots", 0.03),
            telegram_hitl=tg_hitl,
        )

        # Broker / bar fetcher
        if self.synthetic:
            synth = SyntheticBroker()
            self.broker      = synth
            self.bar_fetcher = synth
        else:
            import os
            from src.execution.broker_mt5 import MT5Broker

            self.bar_fetcher = MT5BarFetcher(
                symbol=self.cfg.get("broker", {}).get("symbol", "XAUUSD")
            )
            connected = self.bar_fetcher.connect(
                login    = os.getenv("MT5_LOGIN",    ""),
                password = os.getenv("MT5_PASSWORD", ""),
                server   = os.getenv("MT5_SERVER",   ""),
                path     = os.getenv("MT5_PATH"),
            )
            if not connected:
                return False
            self.broker = MT5Broker(
                symbol=self.cfg.get("broker", {}).get("symbol", "XAUUSD"),
                magic_number=self.cfg.get("broker", {}).get("magic_number", 20260509),
            )
            # MT5Broker reuses the already-initialised connection
            self.broker._mt5 = self.bar_fetcher._mt5
            self.broker._connected = True

        # Telegram
        self.alerter = TelegramAlerter()
        if not self.synthetic:
            self.alerter.alert_startup()

        # Start Telegram commander in background thread
        import threading
        from src.hitl.mt5_interface import TelegramCommander
        self.commander = TelegramCommander(state=self)
        if self.commander._enabled and not self.synthetic:
            t = threading.Thread(target=self.commander.listen, daemon=True)
            t.start()
            logger.info("TelegramCommander: started in background thread")
        else:
            self.commander = None

        logger.info("All components initialised ✓")
        return True

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.running = True
        sig.signal(sig.SIGINT,  lambda s, f: setattr(self, "running", False))
        sig.signal(sig.SIGTERM, lambda s, f: setattr(self, "running", False))

        lot_size       = self.cfg.get("broker", {}).get("lot_size", 0.01)
        bar_interval_s = 60   # M1 = 1 new bar per minute
        last_bar_time  = 0

        while self.running:
            try:
                # ── wait for a new M1 bar ──────────────────────────────────
                bars = self.bar_fetcher.get_m1_bars(n=245)
                if not bars or bars[-1]["time"] == last_bar_time:
                    time.sleep(1)
                    continue
                last_bar_time = bars[-1]["time"]

                # ── remote control checks ─────────────────────────────────
                if self._force_close and self.position is not None:
                    logger.info("TelegramCommander: force-closing position")
                    result = self.broker.close_position(
                        self.position["ticket"], comment="telegram_close"
                    )
                    if result.success:
                        pnl = (result.price - self.position["entry_price"])                               * self.position["dir"] * self.position["lots"] * 100.0
                        self.circuit_breaker.record_trade(pnl)
                        if self.commander:
                            self.commander.send(
                                f"✅ Force closed @ `{result.price:.2f}` | PnL: `${pnl:.2f}`"
                            )
                    self.position    = None
                    self.hold_bars   = 0
                    self._force_close = False

                if self._paused:
                    if self.commander:
                        pass  # status logged per bar only if changed
                    time.sleep(bar_interval_s if not self.synthetic else 0.2)
                    continue

                # ── build features ─────────────────────────────────────────
                features = build_features(bars)
                if features is None:
                    logger.debug(f"Not enough bars yet ({len(bars)}/240)")
                    time.sleep(bar_interval_s)
                    continue

                # ── supervised inference ───────────────────────────────────
                t_inf_start = time.perf_counter()
                probs, conf, sup_lat = run_supervised(self.sup_model, features)
                inf_lat_ms = (time.perf_counter() - t_inf_start) * 1000

                # ── circuit breaker ────────────────────────────────────────
                can_trade, block_reason = self.circuit_breaker.check_can_trade(inf_lat_ms)
                if not can_trade:
                    logger.warning(f"CircuitBreaker: {block_reason}")
                    time.sleep(bar_interval_s)
                    continue

                # ── regime deploy gate ─────────────────────────────────────
                deploy_cfg  = self.cfg.get("deploy_gate", {})
                gate_enabled = deploy_cfg.get("enabled", True)
                gmm2, vol_enc = get_regime(bars, deploy_cfg.get("regime_lookback_bars", 60))
                gated = gate_enabled and (gmm2 == 0.0 and vol_enc == 1.0)
                if gated:
                    logger.info("DeployGate: Bear+HIGH — skipping bar")
                    time.sleep(bar_interval_s)
                    continue

                # ── build RL obs ───────────────────────────────────────────
                closes     = np.array([b["close"] for b in bars])
                ret_1h     = mtf_return_zscored(closes, window=60)
                ret_15m    = mtf_return_zscored(closes, window=15)
                # Scale obs features to match training distribution
                atr_n      = float(np.tanh(_atr_norm(bars) * 30))   # raw ~0.004 → tanh(×30) ≈ 0.12
                trend_n    = _trend_norm(bars)                       # already tanh(×50)
                session_ph = _session_phase(bars)                    # already [0, 0.5, 1.0]

                pos_dir     = 0.0
                unreal_pnl  = 0.0
                hold_frac   = 0.0
                if self.position is not None:
                    pos_dir    = float(self.position["dir"])
                    cur_price  = bars[-1]["close"]
                    # XAUUSD PnL: price_diff × lots × 100 = $ value
                    # (1 standard lot = $100/point → 0.01 lot = $1/point)
                    unreal_pnl = (cur_price - self.position["entry_price"]) \
                                 * self.position["dir"] \
                                 * self.position["lots"] * 100.0
                    hold_frac  = min(1.0, self.hold_bars / self.max_hold)

                # Build base 16D obs; extend to 18D if Phase 9 RL agent loaded
                _rl_obs_16 = build_rl_obs(
                    probs=probs, confidence=conf,
                    position_dir=pos_dir, unrealized_pnl=unreal_pnl,
                    hold_frac=hold_frac,
                    atr_norm=atr_n, trend_norm=trend_n,
                    session_phase=session_ph,
                    regime_quality=gmm2,
                    gs_quartile=0.5,
                    cu_au_regime=0.5,
                    ret_1h=ret_1h, ret_15m=ret_15m,
                )
                if getattr(self, '_rl_obs_dim', 16) == 18:
                    rl_obs = np.concatenate([
                        _rl_obs_16,
                        np.array([session_ph, float(vol_enc)], dtype=np.float32),
                    ])
                else:
                    rl_obs = _rl_obs_16

                # ── RL action ──────────────────────────────────────────────
                size_signal, exit_signal, rl_lat = run_rl(self.rl_agent, rl_obs)
                total_lat = sup_lat + rl_lat

                # ── exit logic ─────────────────────────────────────────────
                EXIT_THRESHOLD = -0.10
                if self.position is not None:
                    self.hold_bars += 1
                    should_exit = (
                        exit_signal < EXIT_THRESHOLD
                        or (not self.synthetic and self.hold_bars >= self.max_hold)
                    )
                    if should_exit:
                        exit_reason = "rl_exit" if exit_signal < EXIT_THRESHOLD \
                                      else "max_hold"
                        dc = DrawdownContext(
                            consecutive_losses=self.circuit_breaker.state.consecutive_losses,
                            daily_pnl_usd=self.circuit_breaker.state.daily_pnl,
                        )
                        ctx = SignalContext(
                            action="CLOSE",
                            confidence=conf,
                            current_price=bars[-1]["close"],
                            entry_price=self.position["entry_price"],
                            unrealized_pnl=unreal_pnl,
                            hold_time_bars=self.hold_bars,
                            exit_reason=exit_reason,
                            position_size_lots=self.position["lots"],
                            drawdown_ctx=dc,
                        )
                        if self.hitl.check_exit(ctx):
                            result = self.broker.close_position(
                                self.position["ticket"], comment=exit_reason
                            )
                            if result.success:
                                pnl = unreal_pnl
                                self.circuit_breaker.record_trade(pnl)
                                self.hitl.record_trade_result(pnl)
                                self.alerter.alert_trade("CLOSE", result.price,
                                                         self.position["lots"], pnl)
                                self._write_log(
                                    bars[-1], "CLOSE",
                                    self.position["dir"], self.position["entry_price"],
                                    result.price, self.position["lots"], pnl,
                                    conf, size_signal, exit_signal, gmm2, vol_enc,
                                    gated, probs,
                                )
                                logger.info(
                                    f"CLOSED | PnL=${pnl:.2f} | reason={exit_reason}"
                                )
                                self.position  = None
                                self.hold_bars = 0

                # ── entry logic ────────────────────────────────────────────
                ENTRY_CONF_GATE  = 0.70
                ENTRY_SIZE_GATE  = 0.10   # |size_signal| must exceed this

                if (self.position is None
                        and conf >= ENTRY_CONF_GATE
                        and abs(size_signal) >= ENTRY_SIZE_GATE):

                    sup_action = int(probs.argmax())  # 0=sell 1=hold 2=buy
                    if sup_action == 1:
                        # supervised says hold — skip
                        pass
                    else:
                        direction = 1 if sup_action == 2 else -1  # +1=long -1=short
                        lots      = self.circuit_breaker.get_position_size(lot_size)
                        ctx = SignalContext(
                            action="BUY" if direction == 1 else "SELL",
                            confidence=conf,
                            current_price=bars[-1]["close"],
                            position_size_lots=lots,
                            regime=f"{'Bull' if gmm2 else 'Bear'}-{'HIGH' if vol_enc else 'LOW'}",
                        )
                        if self.hitl.check_entry(ctx):
                            if direction == 1:
                                result = self.broker.buy(lots, comment="rl_entry")
                            else:
                                result = self.broker.sell(lots, comment="rl_entry")
                            if result.success:
                                self.position  = {
                                    "dir":         direction,
                                    "entry_price": result.price,
                                    "lots":        lots,
                                    "ticket":      result.ticket,
                                }
                                self.hold_bars = 0
                                self._write_log(
                                    bars[-1],
                                    "BUY" if direction == 1 else "SELL",
                                    direction, result.price, result.price,
                                    lots, 0.0, conf, size_signal, exit_signal,
                                    gmm2, vol_enc, gated, probs,
                                )
                                dir_str = "LONG" if direction == 1 else "SHORT"
                                logger.info(
                                    f"ENTRY {dir_str} "
                                    f"@ {result.price:.2f} | conf={conf:.3f} | "
                                    f"rl_size={size_signal:.3f} | lat={total_lat:.1f}ms"
                                )
                                if self.commander:
                                    self.commander.send(
                                        f"✅ *ENTRY {dir_str}* @ `{result.price:.2f}`\n"
                                        f"Lots: `{lots}` | Conf: `{conf:.1%}` | "
                                        f"Regime: `{'Bull' if gmm2 else 'Bear'}-"
                                        f"{'HIGH' if vol_enc else 'LOW'}`"
                                    )

                # ── periodic log ───────────────────────────────────────────
                pos_str = f"pos={'LONG' if self.position else 'FLAT'}"
                logger.info(
                    f"Bar {bars[-1]['time']} | {pos_str} | "
                    f"probs=[{probs[0]:.3f},{probs[1]:.3f},{probs[2]:.3f}] "
                    f"conf={conf:.3f} | rl=[{size_signal:.2f},{exit_signal:.2f}] "
                    f"| gmm2={'Bear' if gmm2==0 else 'Bull'} vol={'HIGH' if vol_enc else 'LOW'}"
                )

                # Wait for next bar
                if not self.synthetic:
                    time.sleep(max(1, bar_interval_s - 5))
                else:
                    time.sleep(0.2)  # slow synthetic enough to observe logs

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(5)

        logger.info(f"Shutdown | HITL stats: {self.hitl.stats}")
        self._log_file.close()
        if not self.synthetic:
            self.alerter.alert_shutdown()

    def _write_log(self, bar, action, direction, entry_price, close_price,
                   lots, pnl, conf, rl_size, rl_exit, gmm2, vol_enc,
                   gated, probs):
        self._log.writerow([
            datetime.now(timezone.utc).isoformat(),
            bar["time"], action, direction,
            round(entry_price, 3), round(close_price, 3),
            lots, round(pnl, 4),
            round(conf, 4), round(rl_size, 4), round(rl_exit, 4),
            gmm2, vol_enc, gated,
            round(probs[0], 4), round(probs[1], 4), round(probs[2], 4),
        ])
        self._log_file.flush()


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HFTExperiment v2 — Phase 8b paper trader")
    parser.add_argument("--config",    default="configs/deployment/production.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run with synthetic price feed (no MT5 required)")
    args = parser.parse_args()

    setup_logger()
    loop = TradingLoop(args.config, synthetic=args.synthetic)
    if loop.initialize():
        loop.run()


if __name__ == "__main__":
    main()