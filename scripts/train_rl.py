#!/usr/bin/env python3
"""Train RL agent on frozen dual-branch supervised model — Phase 3 v2.

v2 improvements (IMPROVEMENTS.md):

§3.2 — Transaction cost curriculum (CCSO):
    Split training into N_EVOLVES stages. Commission and spread anneal
    linearly from 0 at evolve 0 to real values at evolve N_EVOLVES-1.
    A checkpoint is saved at each evolve boundary.
    Low-evolve models: more active, capture small fluctuations.
    High-evolve models: cautious, better cost management.
    Deployment: route by volatility regime (Bear+HIGH → high evolve).

Signal entry bonus/penalty removed (v3 diagnosis):
    Penalty=-1.0 at 99 trades/ep dominated the reward signal, causing
    penalty-avoidance behaviour rather than PnL optimisation.
    Cost curriculum achieves the same goal (teach the agent about costs)
    at the correct reward scale.

Retained from v2:
    - SequenceDataset streaming (no 16.4 GB X array)
    - n_eval_episodes=15
    - confidence_gate=0.70
    - episode_len=8000 (99 trades/ep → 4× learning signal)
    - GMM2 Bear entry gate, G/S max_hold
    - Drive mirroring

Observation (15-dim: +ret_1h obs[13], +ret_15m obs[14]):
    [sell, hold, buy, conf, pos_dir, unreal, hold_t,
     atr_norm, trend_norm, session_phase,
     regime_quality, gs_quartile, cu_au_regime]

Usage:
    python scripts/train_rl.py \\
        --checkpoint models/dual_branch_best.pt \\
        --steps 1500000 --n-evolves 10 \\
        --episode-len 8000 \\
        --eval-every 16000 --n-eval-episodes 15 \\
        --confidence-gate 0.70 \\
        --commission 0.70 --spread-pips 2.0 \\
        --curriculum-warmup 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

import gymnasium as gym
from gymnasium import spaces

from src.data.preprocessing import (
    prepare_features, join_regime_labels,
    get_regime_array, compute_rl_obs_features,
)
from src.data.tick_store import TickStore
from src.encoder.fusion import DualBranchModel
from src.meta_policy.rl_agent import ConfidenceSACAgent
from src.utils.config import set_seed
from src.utils.logger import setup_logger


# ─────────────────────────────────────────────────────────────────────────────
# SequenceDataset
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, seq_len: int):
        assert len(features) > seq_len
        self.features = features
        self.seq_len  = seq_len
        self._n       = len(features) - seq_len

    def __len__(self): return self._n

    def __getitem__(self, i):
        return torch.from_numpy(self.features[i : i + self.seq_len].copy())


# ─────────────────────────────────────────────────────────────────────────────
# Cost curriculum scheduler (§3.2)
# ─────────────────────────────────────────────────────────────────────────────

class CostCurriculum:
    """Anneals transaction costs from 0 to real values over N evolves.

    §3.2 (CCSO): split training into evolves, scale fee rate linearly.
    Qualitatively different behaviour per evolve — save each boundary.

    Args:
        n_evolves:      number of curriculum stages (default 10)
        total_steps:    total RL training steps
        final_commission: real commission in USD (default 0.70)
        final_spread:   real spread in pips (default 2.0)
    """
    def __init__(
        self,
        n_evolves:        int   = 10,
        total_steps:      int   = 1_500_000,
        final_commission: float = 0.70,
        final_spread:     float = 2.0,
    ):
        self.n_evolves        = n_evolves
        self.total_steps      = total_steps
        self.final_commission = final_commission
        self.final_spread     = final_spread
        self.steps_per_evolve = total_steps // n_evolves

    def get_costs(self, step: int) -> tuple[float, float]:
        """Return (commission_usd, spread_pips) for the current step."""
        frac = min(step / self.total_steps, 1.0)
        return (
            frac * self.final_commission,
            frac * self.final_spread,
        )

    def current_evolve(self, step: int) -> int:
        return min(step // self.steps_per_evolve, self.n_evolves - 1)

    def is_evolve_boundary(self, step: int) -> bool:
        return step > 0 and step % self.steps_per_evolve == 0


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class FrozenEncoderEnv(gym.Env):
    """Trading env — v2 with live cost curriculum support.

    Cost curriculum: commission and spread are passed in at each evolve
    boundary by the training loop via update_costs(). The env does not
    manage the curriculum schedule — that lives in the training loop.

    Signal entry bonus/penalty removed (v3 post-mortem):
    At 99 trades/ep, penalty=-1.0 dominated the reward signal and caused
    penalty-avoidance rather than PnL-maximisation. Cost curriculum
    achieves the same pedagogical goal at the correct reward scale.
    """

    _GS_HOLD = {"Q1": 40, "Q23": 60, "Q4": 80}

    def __init__(
        self,
        signals:       np.ndarray,
        confidences:   np.ndarray,
        prices:        np.ndarray,
        atr_norm:      np.ndarray | None = None,
        trend_norm:    np.ndarray | None = None,
        session_phase: np.ndarray | None = None,
        regime_quality: np.ndarray | None = None,
        gs_quartile:    np.ndarray | None = None,
        cu_au_regime:   np.ndarray | None = None,
        gmm2_state:     np.ndarray | None = None,
        ret_1h:         np.ndarray | None = None,   # Phase 6: 60-bar log-return z-score
        ret_15m:        np.ndarray | None = None,   # Phase 6: 15-bar log-return z-score
        inv_pressure:   np.ndarray | None = None,   # Phase 8: inventory_pressure_vio (obs[15])
        max_hold:              int   = 80,
        episode_len:           int   = 8000,
        confidence_gate:       float = 0.70,
        commission_usd:        float = 0.0,    # set by curriculum
        spread_pips:           float = 0.0,    # set by curriculum
        pip_value:             float = 0.10,
        initial_balance:       float = 10_000.0,
        mtm_scale:             float = 0.05,
        hold_penalty_coeff:    float = 0.003,
        early_cut_bonus_frac:  float = 0.40,
    ):
        super().__init__()
        n = len(prices)
        assert len(signals) == len(confidences) == n

        self.signals     = signals.astype(np.float32)
        self.confidences = confidences.astype(np.float32)
        self.prices      = prices.astype(np.float64)

        def _arr(a, default): return a.astype(np.float32) if a is not None else default(n)
        self.atr_norm       = _arr(atr_norm,       lambda n: np.zeros(n, np.float32))
        self.trend_norm     = _arr(trend_norm,      lambda n: np.zeros(n, np.float32))
        self.session_phase  = _arr(session_phase,   lambda n: np.full(n, 0.5, np.float32))
        self.regime_quality = _arr(regime_quality,  lambda n: np.full(n, 0.5, np.float32))
        self.gs_quartile    = _arr(gs_quartile,     lambda n: np.zeros(n, np.float32))
        self.cu_au_regime   = _arr(cu_au_regime,    lambda n: np.full(n, 0.5, np.float32))
        self.gmm2_state     = _arr(gmm2_state,      lambda n: np.ones(n, np.float32))
        # Phase 6 RL additions (validated by Phase 6 report, §10):
        # ret_1h  — 60-bar cumulative log-return z-scored. KS=0.361 (strongest of all candidates),
        #           late-window concentration (SellD[20]=0.605), regime-sensitive (D=0.219).
        # ret_15m — 15-bar cumulative log-return z-scored. KS=0.206, regime-sensitive (D=0.103).
        # Both fail supervised redundancy (Transformer already computes equivalent implicitly)
        # but RL agent operates on obs vector — no access to Transformer internals.
        self.ret_1h         = _arr(ret_1h,         lambda n: np.zeros(n, np.float32))
        self.ret_15m        = _arr(ret_15m,        lambda n: np.zeros(n, np.float32))
        # Phase 8: inventory_pressure_vio — VIO formula, MI=1.92x OHLCV mean.
        # Redundancy ratio=2.62 vs spread_pressure fails supervised threshold,
        # but RL policy never sees spread_pressure directly — only frozen encoder
        # output probs. VIO provides direct inventory-imbalance signal.
        self.inv_pressure   = _arr(inv_pressure,   lambda n: np.zeros(n, np.float32))

        self._max_hold_default = max_hold
        self.episode_len       = min(episode_len, n - 1)
        self.confidence_gate   = confidence_gate
        self.commission        = commission_usd   # updated by curriculum
        self.spread_pips       = spread_pips      # updated by curriculum
        self.pip_value         = pip_value
        self.initial_balance   = initial_balance
        self.mtm_scale         = mtm_scale
        self.hold_penalty_coeff     = hold_penalty_coeff
        self.early_cut_bonus_frac   = early_cut_bonus_frac
        self.exit_threshold: float  = 0.0

        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), np.float32)  # +ret_1h, +ret_15m, +inv_pressure
        self.action_space      = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._offset = 0
        self._reset_state()

    def update_costs(self, commission_usd: float, spread_pips: float) -> None:
        """Called by training loop at each evolve boundary."""
        self.commission  = commission_usd
        self.spread_pips = spread_pips

    @property
    def spread_cost(self): return self.spread_pips * self.pip_value

    def _reset_state(self):
        self.local_idx        = 0
        self.balance          = self.initial_balance
        self.position_dir     = 0
        self.entry_price      = 0.0
        self.entry_conf       = 0.0
        self.hold_time        = 0
        self.n_trades         = 0
        self.episode_pnl      = 0.0
        self.n_wins           = 0
        self._prev_unrealized = 0.0
        self.max_hold         = self._max_hold_default

    def _gi(self): return self._offset + self.local_idx
    def _price(self): return self.prices[self._gi()]

    def _gs_max_hold(self):
        gs = float(self.gs_quartile[self._gi()])
        if gs <= 0.25: return self._GS_HOLD["Q1"]
        if gs >= 0.75: return self._GS_HOLD["Q4"]
        return self._GS_HOLD["Q23"]

    def _unrealized_usd(self):
        if self.position_dir == 0: return 0.0
        return (self._price() - self.entry_price) * self.position_dir / self.pip_value

    def _get_obs(self):
        gi = self._gi()
        sig = self.signals[gi]; conf = self.confidences[gi]
        unreal = np.clip(self._unrealized_usd()/100.0, -5.0, 5.0) if self.position_dir else 0.0
        hold_frac = min(self.hold_time / max(self.max_hold, 1), 1.0)
        return np.array([
            sig[0], sig[1], sig[2], conf,                                   # 0-3
            float(self.position_dir), unreal, hold_frac,                    # 4-6
            self.atr_norm[gi], self.trend_norm[gi], self.session_phase[gi], # 7-9
            self.regime_quality[gi], self.gs_quartile[gi], self.cu_au_regime[gi],  # 10-12
            self.ret_1h[gi], self.ret_15m[gi],                              # 13-14 (Phase 6)
            self.inv_pressure[gi],                                          # 15    (Phase 8)
        ], dtype=np.float32)

    def _close_position(self, voluntary=False):
        if self.position_dir == 0: return 0.0
        raw_pnl = (self._price() - self.entry_price) * self.position_dir / self.pip_value
        pnl_usd = raw_pnl - self.spread_cost / self.pip_value * 0.5 - self.commission
        self.balance += pnl_usd; self.episode_pnl += pnl_usd
        self.n_trades += 1
        if pnl_usd > 0: self.n_wins += 1
        bonus = 0.0
        if voluntary and pnl_usd < 0 and self.hold_time < self.max_hold * 0.8:
            avoided = (abs(pnl_usd)/max(self.hold_time,1)) * (self.max_hold-self.hold_time)
            bonus = min(avoided * self.early_cut_bonus_frac, abs(pnl_usd) * 0.5)
        conf = self.entry_conf
        gated = 0.0 if conf <= self.confidence_gate else pnl_usd * (conf-self.confidence_gate)/(1.0-self.confidence_gate)
        gated += bonus
        self.position_dir = 0; self.entry_price = 0.0
        self.entry_conf = 0.0; self.hold_time = 0; self._prev_unrealized = 0.0
        return gated

    def step(self, action):
        gi = self._gi()
        pos_action = float(action[0]); exit_logit = float(action[1]) if len(action)>1 else 0.0
        reward = 0.0; should_exit = exit_logit < self.exit_threshold
        self.max_hold = self._gs_max_hold()
        if self.position_dir != 0 and self.hold_time >= self.max_hold:
            reward += self._close_position(voluntary=False)
        if should_exit and self.position_dir != 0:
            reward += self._close_position(voluntary=True)
        in_bull = float(self.gmm2_state[gi]) > 0.5
        if not should_exit:
            if pos_action > 0.0 and self.position_dir != 1:
                if self.position_dir == -1: reward += self._close_position(voluntary=True)
                if self.position_dir == 0 and in_bull:
                    self.position_dir = 1; self.entry_price = self._price() + self.spread_cost*0.5
                    self.entry_conf = self.confidences[gi]; self.hold_time = 0; self._prev_unrealized = 0.0
            elif pos_action < 0.0 and self.position_dir != -1:
                if self.position_dir == 1: reward += self._close_position(voluntary=True)
                if self.position_dir == 0 and in_bull:
                    self.position_dir = -1; self.entry_price = self._price() - self.spread_cost*0.5
                    self.entry_conf = self.confidences[gi]; self.hold_time = 0; self._prev_unrealized = 0.0
        if self.position_dir != 0:
            self.hold_time += 1
            cur_unreal = self._unrealized_usd()
            reward += (cur_unreal - self._prev_unrealized) * self.mtm_scale
            self._prev_unrealized = cur_unreal
            reward -= self.hold_penalty_coeff * (self.hold_time/max(self.max_hold,1))**2
        self.local_idx += 1
        terminated = self.local_idx >= self.episode_len
        truncated  = self.balance <= 0
        if terminated or truncated:
            if self.position_dir != 0: reward += self._close_position(voluntary=False)
        obs = self._get_obs() if not (terminated or truncated) else np.zeros(16, np.float32)
        return obs, reward, terminated, truncated, {
            "balance": self.balance, "episode_pnl": self.episode_pnl,
            "n_trades": self.n_trades, "n_wins": self.n_wins,
            "win_rate": self.n_wins/max(self.n_trades,1), "max_hold": self.max_hold,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_state()
        max_off = len(self.prices) - self.episode_len - 1
        self._offset = np.random.randint(0, max_off) if max_off > 0 else 0
        return self._get_obs(), {}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_model_features(model, features, seq_len, device, batch_size=2048):
    ds = SequenceDataset(features, seq_len)
    n_cpu  = min(8, __import__("os").cpu_count() or 1)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=n_cpu,
                        pin_memory=True, shuffle=False, persistent_workers=True,
                        prefetch_factor=4)
    all_probs, all_confs = [], []
    model.eval()
    logger.info(f"Extracting signals across {len(loader)} batches...")
    with torch.no_grad():
        for i, bx in enumerate(loader):
            bx = bx.to(device)
            logits, conf = model(bx, None)
            all_probs.append(torch.softmax(logits,-1).cpu().numpy())
            all_confs.append(conf.squeeze(-1).cpu().numpy())
            if i % 500 == 0: logger.info(f"Processed {i}/{len(loader)} batches...")
    signals = np.concatenate(all_probs); confidences = np.concatenate(all_confs)
    argmax = signals.argmax(1)
    logger.info(f"Features: {signals.shape} | conf {confidences.mean():.3f}±{confidences.std():.3f} | "
                f"sell={np.mean(argmax==0):.1%} hold={np.mean(argmax==1):.1%} buy={np.mean(argmax==2):.1%}")
    return signals, confidences


def evaluate(agent, env, n_episodes=15):
    total_pnl=0.0; total_trades=0; total_wins=0
    for _ in range(n_episodes):
        obs, _ = env.reset(); done = False
        while not done:
            pa, se = agent.select_action(obs, confidence=obs[3], eval_mode=True)
            obs, _, te, tr, info = env.step(np.array([pa, -1.0 if se else 0.5], dtype=np.float32))
            done = te or tr
        total_pnl += info["episode_pnl"]; total_trades += info["n_trades"]; total_wins += info["n_wins"]
    return total_pnl/n_episodes, total_trades/n_episodes, total_wins/max(total_trades,1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3 RL v2 — cost curriculum")
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--symbol",      default="XAUUSD")
    parser.add_argument("--checkpoint",  default="models/dual_branch_best.pt")
    parser.add_argument("--steps",  type=int,   default=1_500_000)
    parser.add_argument("--n-evolves",type=int,  default=10,
                        help="Cost curriculum stages (0=free, N=full cost)")
    parser.add_argument("--episode-len",type=int,default=8000)
    parser.add_argument("--confidence-gate",type=float,default=0.70)
    parser.add_argument("--commission", type=float,default=0.70)
    parser.add_argument("--spread-pips",type=float,default=2.0)
    parser.add_argument("--n-eval-episodes",type=int,default=15)
    parser.add_argument("--eval-every",  type=int, default=16_000)
    parser.add_argument("--seq-length",  type=int, default=120)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--max-hold",    type=int, default=80)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--save-dir",    default="models")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--mtm-scale",        type=float,default=0.05)
    parser.add_argument("--hold-penalty",     type=float,default=0.003)
    parser.add_argument("--early-cut-bonus",  type=float,default=0.40)
    parser.add_argument("--curriculum-warmup",type=int,  default=100_000)
    parser.add_argument("--batch-size",       type=int,  default=8192)  # A100
    parser.add_argument("--regime-csv",       default="data/regime/daily_regime_labels.csv")
    # Option B: extended flat-cost hardening
    parser.add_argument("--resume-agent",     default=None,
                        help="Path to existing SAC checkpoint (.pt) to resume from")
    parser.add_argument("--flat-costs",       action="store_true",
                        help="Disable curriculum — run all steps at full --commission/--spread-pips")
    args = parser.parse_args()

    setup_logger(); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 on A100

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}"); sys.exit(1)

    # Fast path: precomputed .npz (~8 s) vs full pipeline (~10 min)
    ws = args.window_size
    _ready = Path("/content/drive/MyDrive/Colab Notebooks/training_ready.npz")
    if _ready.exists():
        logger.info(f"Loading precomputed training_ready: {_ready}")
        _d           = np.load(str(_ready), allow_pickle=True)
        features     = _d["features"]       # already trimmed by ws
        close_prices = _d["close"]
        # Reconstruct 6-col regime array from individual arrays
        regime_arr   = np.stack([
            _d["gmm2"], _d["km_enc"], _d["vol_enc"],
            _d["gs_q"], _d["cu_au"], _d["rq"]
        ], axis=1).astype(np.float32)
        logger.info(f"  Loaded {len(features):,} sequences from cache")
    else:
        logger.info("training_ready.npz not found — running full pipeline")
        store = TickStore(f"{args.data_dir}/ticks.duckdb")
        df    = store.query_ohlcv(args.symbol, "M1"); store.close()
        if df.is_empty(): logger.error("No data."); sys.exit(1)
        df = join_regime_labels(df, args.regime_csv)
        regime_arr_full = get_regime_array(df)
        features, close_prices, _, _ = prepare_features(df, window_size=ws)
        features = features[ws:]; close_prices = close_prices[ws:]
        regime_arr = regime_arr_full[ws:]

    sl = args.seq_length; n_seq = len(features) - sl
    seq_prices = close_prices[sl-1:sl-1+n_seq]; seq_regime = regime_arr[sl-1:sl-1+n_seq]
    seq_gmm2=seq_regime[:,0]; seq_gs=seq_regime[:,3]; seq_cu=seq_regime[:,4]; seq_rq=seq_regime[:,5]
    logger.info(f"Sequences: {n_seq:,} | Bull={seq_gmm2.mean():.1%}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(ckpt.get("config", {}))
    model = DualBranchModel.from_config(cfg.model) if cfg else DualBranchModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad = False
    _np = sum(p.numel() for p in model.parameters())
    logger.info(f"Frozen model: {_np:,} params")  # torch.compile skipped

    signals, confidences = extract_model_features(model, features, sl, device, args.batch_size)

    try:
        raw_ts = df["timestamp"].to_list() if "timestamp" in df.columns else None
        seq_ts = raw_ts[ws+sl-1:ws+sl-1+len(signals)] if raw_ts else None
        if seq_ts and len(seq_ts) < len(signals): seq_ts = None
    except Exception: seq_ts = None

    atr_norm, trend_norm, session_phase = compute_rl_obs_features(seq_prices, seq_ts)

    # ── Phase 6 RL additions: ret_1h and ret_15m ─────────────────────────────
    # Computed from seq_prices — no NPZ rebuild required.
    # Phase 6 report §10: both fail supervised redundancy (Transformer computes
    # equivalent implicitly) but RL agent operates on obs vector directly and
    # has no access to Transformer internals. ret_1h has KS=0.361 (project best),
    # regime-sensitive (D=0.219). ret_15m KS=0.206, regime-sensitive (D=0.103).
    def _mtf_return_zscored(prices: np.ndarray, window: int) -> np.ndarray:
        """Cumulative log-return over `window` bars, 120-bar rolling z-score, tanh-clipped."""
        log_ret = np.concatenate([[0.0], np.log(prices[1:] / (prices[:-1] + 1e-8))])
        rolling = np.convolve(log_ret, np.ones(window), mode='same')
        out     = np.empty(len(rolling), dtype=np.float32)
        W       = 120
        for i in range(len(rolling)):
            lo_w = max(0, i - W + 1)
            win  = rolling[lo_w:i + 1]
            mu, sig = win.mean(), win.std()
            out[i] = float(np.tanh((rolling[i] - mu) / (sig + 1e-8)))
        return out

    logger.info("Computing ret_1h and ret_15m for RL obs extension...")
    ret_1h_arr  = _mtf_return_zscored(seq_prices, window=60)
    ret_15m_arr = _mtf_return_zscored(seq_prices, window=15)
    logger.info(f"ret_1h: mean={ret_1h_arr.mean():.4f} std={ret_1h_arr.std():.4f}")
    logger.info(f"ret_15m: mean={ret_15m_arr.mean():.4f} std={ret_15m_arr.std():.4f}")

    # ── Phase 8: inventory_pressure_vio ──────────────────────────────────────
    # VIO[t] = Σ(V_k × sign_k) / Σ(V_k)  over rolling W=20 bars
    # Phase 8 results: KS=0.112, MI=0.01076 (1.92x OHLCV mean), Red=2.62 (RL safe).
    # tick_volume_raw from NPZ if available; falls back to uniform volume (VIO→mean direction).
    logger.info("Computing inventory_pressure_vio (Phase 8 obs[15])...")
    W_vio = 20
    try:
        tick_vol = _d["tick_volume_raw"].astype(np.float64)
        logger.info("tick_volume_raw found in NPZ — using true VIO")
    except Exception:
        tick_vol = np.ones(len(seq_prices), dtype=np.float64)
        tick_vol = np.ones(len(seq_prices), dtype=np.float64)
        logger.warning("tick_volume_raw not in NPZ — using uniform volume (VIO≈mean direction)")
    bar_dir = np.sign(seq_prices - np.concatenate([[seq_prices[0]], seq_prices[:-1]])).astype(np.float64)
    inv_pressure_arr = np.empty(len(seq_prices), dtype=np.float32)
    for i in range(len(seq_prices)):
        lo  = max(0, i - W_vio + 1)
        num = np.sum(tick_vol[lo:i+1] * bar_dir[lo:i+1])
        den = np.sum(tick_vol[lo:i+1]) + 1e-8
        inv_pressure_arr[i] = float(num / den)
    logger.info(f"inv_pressure: mean={inv_pressure_arr.mean():.4f} std={inv_pressure_arr.std():.4f}")

    split = int(len(signals) * 0.8)
    logger.info(f"Train={split:,} Eval={len(signals)-split:,}")

    curriculum = CostCurriculum(
        n_evolves=args.n_evolves, total_steps=args.steps,
        final_commission=args.commission, final_spread=args.spread_pips,
    )
    if args.flat_costs:
        # Option B: skip curriculum — start at full costs immediately
        logger.info(f"--flat-costs: curriculum disabled. "
                    f"Fixed comm=${args.commission} spread={args.spread_pips}pip from step 1")
        args.n_evolves = 1   # save one checkpoint at end

    def make_env(lo, hi, comm=0.0, sp=0.0):
        # flat-costs: always use full costs for train env too
        if args.flat_costs:
            comm = args.commission; sp = args.spread_pips
        return FrozenEncoderEnv(
            signals=signals[lo:hi], confidences=confidences[lo:hi],
            prices=seq_prices[lo:hi], atr_norm=atr_norm[lo:hi],
            trend_norm=trend_norm[lo:hi], session_phase=session_phase[lo:hi],
            regime_quality=seq_rq[lo:hi], gs_quartile=seq_gs[lo:hi],
            cu_au_regime=seq_cu[lo:hi], gmm2_state=seq_gmm2[lo:hi],
            ret_1h=ret_1h_arr[lo:hi], ret_15m=ret_15m_arr[lo:hi],
            inv_pressure=inv_pressure_arr[lo:hi],
            max_hold=args.max_hold, episode_len=args.episode_len,
            confidence_gate=args.confidence_gate,
            commission_usd=comm, spread_pips=sp,
            mtm_scale=args.mtm_scale, hold_penalty_coeff=args.hold_penalty,
            early_cut_bonus_frac=args.early_cut_bonus,
        )

    train_env = make_env(0, split)
    eval_env  = make_env(split, len(signals), args.commission, args.spread_pips)

    agent = ConfidenceSACAgent(
        obs_dim=16, hidden_dims=[512, 512],  # +ret_1h, +ret_15m, +inv_pressure
        device=str(device), curriculum_warmup_steps=args.curriculum_warmup,
        buffer_capacity=2_000_000, batch_size=2048,
    )
    if args.resume_agent:
        logger.info(f"Option B: resuming agent from {args.resume_agent}")
        rl_ckpt = torch.load(args.resume_agent, map_location=device)

        def _pad_state_dict(sd, new_obs_dim):
            """Pad first-layer weight if obs_dim increased. New dims init to zero
            so new features start as no-ops — policy learns their value from experience."""
            sd = {k: v.clone() for k, v in sd.items()}
            first_w_key = "0.weight"   # Sequential actor/critic: layer 0
            if first_w_key in sd:
                old_w = sd[first_w_key]          # (hidden, old_obs)
                old_in = old_w.shape[1]
                if old_in < new_obs_dim:
                    pad = torch.zeros(old_w.shape[0], new_obs_dim - old_in,
                                      dtype=old_w.dtype, device=old_w.device)
                    sd[first_w_key] = torch.cat([old_w, pad], dim=1)
                    logger.info(f"  Padded {first_w_key}: {old_in}→{new_obs_dim} input dims")
            return sd

        obs_dim = 16
        actor_sd  = _pad_state_dict(rl_ckpt["actor"],  obs_dim)
        critic1_sd = _pad_state_dict(rl_ckpt["critic1"], obs_dim + 2)
        critic2_sd = _pad_state_dict(rl_ckpt["critic2"], obs_dim + 2)
        agent.actor.load_state_dict(actor_sd)
        agent.critic1.load_state_dict(critic1_sd)
        agent.critic2.load_state_dict(critic2_sd)
        agent.target_critic1.load_state_dict(critic1_sd)
        agent.target_critic2.load_state_dict(critic2_sd)
        logger.info("  Agent weights loaded with zero-padded obs dim (replay buffer reset)")
    mode_str = "flat-cost hardening" if args.flat_costs else f"cost curriculum n_evolves={args.n_evolves}"
    logger.info(
        f"RL Phase 5: {mode_str}\n"
        f"  commission: ${args.commission}  spread: {args.spread_pips}pips\n"
        f"  episode_len={args.episode_len} | confidence_gate={args.confidence_gate}"
    )

    obs, _ = train_env.reset()
    best_eval_pnl = -float("inf"); ep_rewards=[]; ep_pnls=[]; ep_reward=0.0; ep_count=0
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps+1):
        agent.set_step(step)
        train_env.exit_threshold = agent._exit_threshold

        # Cost curriculum update
        comm, sp = curriculum.get_costs(step)
        train_env.update_costs(comm, sp)

        # Save evolve checkpoint
        if curriculum.is_evolve_boundary(step):
            ev = curriculum.current_evolve(step)
            agent.save(str(save_dir / f"rl_agent_evolve{ev}.pt"))
            logger.info(f"  Evolve {ev} checkpoint saved (comm=${comm:.2f} spread={sp:.1f}pip)")

        pa, se = agent.select_action(obs, confidence=float(obs[3]))
        action = np.array([pa, -1.0 if se else 0.5], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        agent.store(obs, action, reward, next_obs, done)
        ep_reward += reward; obs = next_obs
        if agent.buffer_size >= agent.batch_size: agent.update()
        if done:
            ep_rewards.append(ep_reward); ep_pnls.append(info["episode_pnl"])
            ep_count += 1; ep_reward = 0.0; obs, _ = train_env.reset()

        if step % args.eval_every == 0:
            avg_r   = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
            avg_pnl = np.mean(ep_pnls[-20:])    if ep_pnls    else 0.0
            ev_pnl, ev_tr, ev_wr = evaluate(agent, eval_env, args.n_eval_episodes)
            evolve = curriculum.current_evolve(step)
            logger.info(
                f"Step {step:,}/{args.steps:,} (ep={ep_count} evolve={evolve}) | "
                f"costs: comm=${comm:.2f} sp={sp:.1f}pip | "
                f"Train r={avg_r:.2f} pnl=${avg_pnl:.2f} | "
                f"Eval({args.n_eval_episodes}ep) pnl=${ev_pnl:.2f} tr={ev_tr:.0f} wr={ev_wr:.1%}"
            )
            if ev_pnl > best_eval_pnl:
                best_eval_pnl = ev_pnl
                agent.save(str(save_dir / "rl_agent_best.pt"))
                logger.info(f"  → Best eval PnL: ${ev_pnl:.2f}")

    final_pnl, final_tr, final_wr = evaluate(agent, eval_env, args.n_eval_episodes*2)
    logger.info(
        f"\n{'='*60}\nPHASE 3 RL v2 COMPLETE\n{'='*60}\n"
        f"Steps: {args.steps:,} | Episodes: {ep_count}\n"
        f"Best eval PnL: ${best_eval_pnl:.2f}\n"
        f"Final eval PnL: ${final_pnl:.2f}\n"
        f"Final trades: {final_tr:.0f} | WR: {final_wr:.1%}\n"
        f"Evolve checkpoints saved: {args.n_evolves}\n{'='*60}"
    )

if __name__ == "__main__":
    main()