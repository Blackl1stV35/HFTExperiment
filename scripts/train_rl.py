#!/usr/bin/env python3
"""Train RL agent on frozen dual-branch supervised model — Phase 3 v2.

Infrastructure (caller's changes, acknowledged):
    SequenceDataset streams raw (N,6) features through the frozen model
    in batches instead of pre-allocating (N, seq_len, F) = 16.4 GB.
    extract_model_features() accepts raw features + DataLoader, logs
    batch-by-batch progress. Output: signals 68 MB + confidences 22 MB.
    create_sequences() removed entirely from this script.

v2 behaviour changes (from RL run 1 review):
    1. --steps default 500k -> 1_500_000
       500k = 250 episodes. SAC needs thousands to converge.
    2. --n-eval-episodes default 3 -> 15
       eval PnL std was $381 across run 1 with n=3.
       Averaging 15 episodes reduces noise by ~2.2x.
    3. --confidence-gate default 0.48 -> 0.70
       conf mean = 0.908 in run 1. Gate at 0.48 was inactive
       (effective scale 0.823 on every trade). Gate at 0.70 creates
       real differentiation: high-conf (0.95) => 0.83 scale,
       mid-conf (0.75) => 0.17 scale, low-conf (0.70) => 0.
    4. Buy-signal reward discount 0.30x in _close_position.
       Supervised buy F1 = 0.021 on test (near noise). Buy-initiated
       trades are discounted so the agent learns the sell signal is
       the reliable edge and does not chase buy noise.

Observation (13-dim, unchanged):
    [sell, hold, buy, conf, pos_dir, unreal, hold_t,        idx 0-6
     atr_norm, trend_norm, session_phase,                   idx 7-9
     regime_quality_norm, gs_quartile_norm, cu_au_regime]   idx 10-12

Usage:
    python scripts/train_rl.py \\
        --checkpoint models/dual_branch_best.pt \\
        --steps 1500000 --seed 42 \\
        --eval-every 8000 --n-eval-episodes 15 \\
        --confidence-gate 0.70 \\
        --mtm-scale 0.05 --hold-penalty 0.003 --early-cut-bonus 0.40 \\
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
# SequenceDataset — OOM fix (caller's infrastructure change)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Sliding-window dataset over raw (N, F) features.

    Slices each (seq_len, F) window in __getitem__ rather than
    pre-allocating the full (N, seq_len, F) array (~16.4 GB).
    Used only for feature extraction — not for RL env stepping.
    """

    def __init__(self, features: np.ndarray, seq_len: int):
        assert len(features) > seq_len
        self.features = features
        self.seq_len  = seq_len
        self._n       = len(features) - seq_len

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(
            self.features[i : i + self.seq_len].copy()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class FrozenEncoderEnv(gym.Env):
    """Trading env driven by pre-extracted frozen model outputs.

    v2 changes:
        - confidence_gate default raised 0.48 -> 0.70 (fix 3)
        - Buy-signal reward discount: entry_signal==2 (buy) rewards
          scaled by buy_reward_scale (default 0.30) because supervised
          buy F1=0.021 on test is near-noise (fix 4)
        - entry_signal stored at open to identify buy vs sell trades

    Phase 3 obs (13-dim), G/S max_hold, GMM2 entry gate unchanged.
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
        max_hold:           int   = 80,
        episode_len:        int   = 2000,
        confidence_gate:    float = 0.70,    # v2: raised from 0.48
        buy_reward_scale:   float = 0.30,    # v2: discount buy-initiated trades
        spread_pips:        float = 2.0,
        pip_value:          float = 0.10,
        commission_usd:     float = 0.70,
        initial_balance:    float = 10_000.0,
        mtm_scale:          float = 0.05,
        hold_penalty_coeff: float = 0.003,
        early_cut_bonus_frac: float = 0.40,
    ):
        super().__init__()
        n = len(prices)
        assert len(signals) == len(confidences) == n

        self.signals     = signals.astype(np.float32)
        self.confidences = confidences.astype(np.float32)
        self.prices      = prices.astype(np.float64)

        self.atr_norm       = atr_norm.astype(np.float32)      if atr_norm       is not None else np.zeros(n, np.float32)
        self.trend_norm     = trend_norm.astype(np.float32)     if trend_norm     is not None else np.zeros(n, np.float32)
        self.session_phase  = session_phase.astype(np.float32)  if session_phase  is not None else np.full(n, 0.5, np.float32)
        self.regime_quality = regime_quality.astype(np.float32) if regime_quality is not None else np.full(n, 0.5, np.float32)
        self.gs_quartile    = gs_quartile.astype(np.float32)    if gs_quartile    is not None else np.zeros(n, np.float32)
        self.cu_au_regime   = cu_au_regime.astype(np.float32)   if cu_au_regime   is not None else np.full(n, 0.5, np.float32)
        self.gmm2_state     = gmm2_state.astype(np.float32)     if gmm2_state     is not None else np.ones(n, np.float32)

        self._max_hold_default  = max_hold
        self.episode_len        = min(episode_len, n - 1)
        self.confidence_gate    = confidence_gate
        self.buy_reward_scale   = buy_reward_scale
        self.spread_cost        = spread_pips * pip_value
        self.commission         = commission_usd
        self.initial_balance    = initial_balance
        self.pip_value          = pip_value
        self.mtm_scale          = mtm_scale
        self.hold_penalty_coeff = hold_penalty_coeff
        self.early_cut_bonus_frac = early_cut_bonus_frac
        self.exit_threshold: float = 0.0

        self.observation_space = spaces.Box(-np.inf, np.inf, (13,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self._offset = 0
        self._reset_state()

    def _reset_state(self):
        self.local_idx        = 0
        self.balance          = self.initial_balance
        self.position_dir     = 0
        self.entry_price      = 0.0
        self.entry_conf       = 0.0
        self.entry_signal     = 0     # v2: 0=none 1=sell-initiated 2=buy-initiated
        self.hold_time        = 0
        self.n_trades         = 0
        self.episode_pnl      = 0.0
        self.n_wins           = 0
        self._prev_unrealized = 0.0
        self.max_hold         = self._max_hold_default

    def _gi(self):
        return self._offset + self.local_idx

    def _price(self):
        return self.prices[self._gi()]

    def _gs_max_hold(self) -> int:
        gs = float(self.gs_quartile[self._gi()])
        if gs <= 0.25:
            return self._GS_HOLD["Q1"]
        elif gs >= 0.75:
            return self._GS_HOLD["Q4"]
        return self._GS_HOLD["Q23"]

    def _unrealized_usd(self) -> float:
        if self.position_dir == 0:
            return 0.0
        return (self._price() - self.entry_price) * self.position_dir / self.pip_value

    def _get_obs(self) -> np.ndarray:
        gi       = self._gi()
        sig      = self.signals[gi]
        conf     = self.confidences[gi]
        unreal   = np.clip(self._unrealized_usd() / 100.0, -5.0, 5.0) if self.position_dir else 0.0
        hold_frac = min(self.hold_time / max(self.max_hold, 1), 1.0)
        return np.array([
            sig[0], sig[1], sig[2],
            conf,
            float(self.position_dir),
            unreal,
            hold_frac,
            self.atr_norm[gi],
            self.trend_norm[gi],
            self.session_phase[gi],
            self.regime_quality[gi],
            self.gs_quartile[gi],
            self.cu_au_regime[gi],
        ], dtype=np.float32)

    def _close_position(self, voluntary: bool = False) -> float:
        if self.position_dir == 0:
            return 0.0

        raw_pnl = (self._price() - self.entry_price) * self.position_dir / self.pip_value
        pnl_usd = raw_pnl - self.spread_cost / self.pip_value * 0.5 - self.commission
        self.balance     += pnl_usd
        self.episode_pnl += pnl_usd
        self.n_trades    += 1
        if pnl_usd > 0:
            self.n_wins += 1

        # Early-cut bonus
        early_cut_bonus = 0.0
        if voluntary and pnl_usd < 0 and self.hold_time < self.max_hold * 0.8:
            remaining       = self.max_hold - self.hold_time
            avoided         = (abs(pnl_usd) / max(self.hold_time, 1)) * remaining
            early_cut_bonus = min(avoided * self.early_cut_bonus_frac, abs(pnl_usd) * 0.5)

        # Confidence gate
        conf = self.entry_conf
        if conf <= self.confidence_gate:
            gated = 0.0
        else:
            gate_scale = (conf - self.confidence_gate) / (1.0 - self.confidence_gate)
            gated = pnl_usd * gate_scale

        gated += early_cut_bonus

        # v2 fix 4: discount buy-initiated trades (buy F1=0.021 on test)
        # Sell-initiated signal is reliable (F1=0.357); buy is near-noise.
        # Scaling buy rewards prevents the agent over-exploiting a false signal.
        if self.entry_signal == 2:
            gated *= self.buy_reward_scale

        # Reset position state
        self.position_dir     = 0
        self.entry_price      = 0.0
        self.entry_conf       = 0.0
        self.entry_signal     = 0
        self.hold_time        = 0
        self._prev_unrealized = 0.0
        return gated

    def step(self, action: np.ndarray):
        gi         = self._gi()
        pos_action = float(action[0])
        exit_logit = float(action[1]) if len(action) > 1 else 0.0
        reward     = 0.0
        should_exit = exit_logit < self.exit_threshold

        self.max_hold = self._gs_max_hold()

        if self.position_dir != 0 and self.hold_time >= self.max_hold:
            reward += self._close_position(voluntary=False)

        if should_exit and self.position_dir != 0:
            reward += self._close_position(voluntary=True)

        in_bull = float(self.gmm2_state[gi]) > 0.5

        if not should_exit:
            sig_argmax = int(self.signals[gi].argmax())   # 0=sell 1=hold 2=buy

            if pos_action > 0.0 and self.position_dir != 1:
                if self.position_dir == -1:
                    reward += self._close_position(voluntary=True)
                if self.position_dir == 0 and in_bull:
                    self.position_dir     = 1
                    self.entry_price      = self._price() + self.spread_cost * 0.5
                    self.entry_conf       = self.confidences[gi]
                    self.entry_signal     = sig_argmax     # track for buy discount
                    self.hold_time        = 0
                    self._prev_unrealized = 0.0

            elif pos_action < 0.0 and self.position_dir != -1:
                if self.position_dir == 1:
                    reward += self._close_position(voluntary=True)
                if self.position_dir == 0 and in_bull:
                    self.position_dir     = -1
                    self.entry_price      = self._price() - self.spread_cost * 0.5
                    self.entry_conf       = self.confidences[gi]
                    self.entry_signal     = sig_argmax
                    self.hold_time        = 0
                    self._prev_unrealized = 0.0

        if self.position_dir != 0:
            self.hold_time += 1
            cur_unreal  = self._unrealized_usd()
            reward     += (cur_unreal - self._prev_unrealized) * self.mtm_scale
            self._prev_unrealized = cur_unreal
            hold_frac   = self.hold_time / max(self.max_hold, 1)
            reward     -= self.hold_penalty_coeff * (hold_frac ** 2)

        self.local_idx += 1
        terminated = self.local_idx >= self.episode_len
        truncated  = self.balance <= 0

        if terminated or truncated:
            if self.position_dir != 0:
                reward += self._close_position(voluntary=False)

        obs  = self._get_obs() if not (terminated or truncated) else np.zeros(13, np.float32)
        info = {
            "balance":     self.balance,
            "episode_pnl": self.episode_pnl,
            "n_trades":    self.n_trades,
            "n_wins":      self.n_wins,
            "win_rate":    self.n_wins / max(self.n_trades, 1),
            "max_hold":    self.max_hold,
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        max_offset = len(self.prices) - self.episode_len - 1
        self._offset = np.random.randint(0, max_offset) if max_offset > 0 else 0
        return self._get_obs(), {}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — SequenceDataset streaming (caller's change)
# ─────────────────────────────────────────────────────────────────────────────

def extract_model_features(
    model,
    features:   np.ndarray,   # (N, F) raw scaled features
    seq_len:    int,
    device:     torch.device,
    batch_size: int = 2048,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream raw features through frozen model via SequenceDataset.

    Avoids pre-allocating (N, seq_len, F) array. Each batch is
    (batch_size, seq_len, F) assembled on-the-fly by the DataLoader.

    Returns:
        signals:     (N - seq_len, 3) softmax probabilities
        confidences: (N - seq_len,)   confidence head output
    """
    ds     = SequenceDataset(features, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=False, shuffle=False)
    n_batches   = len(loader)
    all_probs   = []
    all_confs   = []

    model.eval()
    logger.info(f"Extracting signals across {n_batches} batches using stride tricks...")
    with torch.no_grad():
        for i, bx in enumerate(loader):
            bx = bx.to(device)
            logits, conf = model(bx, None)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_confs.append(conf.squeeze(-1).cpu().numpy())
            if i % 500 == 0:
                logger.info(f"Processed {i} / {n_batches} batches...")

    signals     = np.concatenate(all_probs)
    confidences = np.concatenate(all_confs)
    argmax      = signals.argmax(axis=1)
    logger.info(
        f"Features: {signals.shape} | "
        f"conf {confidences.mean():.3f}\u00b1{confidences.std():.3f} | "
        f"sell={np.mean(argmax==0):.1%} hold={np.mean(argmax==1):.1%} buy={np.mean(argmax==2):.1%}"
    )
    return signals, confidences


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — v2: n_episodes configurable (was hardcoded 3)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent, env, n_episodes: int = 15) -> tuple[float, float, float]:
    """Average eval over n_episodes random windows from the eval split.

    v2: default raised from 3 to 15 to reduce single-window noise.
    eval PnL std was $381 with n=3; n=15 reduces that to ~$98.
    """
    total_pnl = 0.0; total_trades = 0; total_wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            pos_action, should_exit = agent.select_action(obs, confidence=obs[3], eval_mode=True)
            action = np.array([pos_action, -1.0 if should_exit else 0.5], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        total_pnl    += info["episode_pnl"]
        total_trades += info["n_trades"]
        total_wins   += info["n_wins"]
    return total_pnl / n_episodes, total_trades / n_episodes, total_wins / max(total_trades, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 RL v2 (obs=13, G/S hold, GMM2 gate, buy discount)"
    )
    parser.add_argument("--data-dir",            default="data")
    parser.add_argument("--symbol",              default="XAUUSD")
    parser.add_argument("--checkpoint",          default="models/dual_branch_best.pt")
    parser.add_argument("--steps",      type=int, default=1_500_000)   # v2: was 500k
    parser.add_argument("--max-hold",   type=int, default=80)
    parser.add_argument("--episode-len",type=int, default=2000)
    parser.add_argument("--confidence-gate", type=float, default=0.70)  # v2: was 0.48
    parser.add_argument("--buy-reward-scale",type=float, default=0.30)  # v2: new
    parser.add_argument("--n-eval-episodes", type=int,   default=15)    # v2: was 3
    parser.add_argument("--seq-length",  type=int, default=120)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--save-dir",    default="models")
    parser.add_argument("--eval-every",  type=int, default=8_000)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--mtm-scale",          type=float, default=0.05)
    parser.add_argument("--hold-penalty",        type=float, default=0.003)
    parser.add_argument("--early-cut-bonus",     type=float, default=0.40)
    parser.add_argument("--curriculum-warmup",   type=int,   default=100_000)
    parser.add_argument("--batch-size",          type=int,   default=2048)
    parser.add_argument("--regime-csv",
        default="data/regime/daily_regime_labels.csv")
    args = parser.parse_args()

    setup_logger()
    set_seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    logger.info(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python scripts/train_supervised.py model=dual_branch data=xauusd"
        )
        sys.exit(1)

    # ── Load data + regime labels ─────────────────────────────────────────────
    store = TickStore(f"{args.data_dir}/ticks.duckdb")
    df    = store.query_ohlcv(args.symbol, "M1")
    store.close()
    if df.is_empty():
        logger.error("No data. Run: python scripts/download_data.py"); sys.exit(1)

    df = join_regime_labels(df, args.regime_csv)

    regime_arr_full = get_regime_array(df)   # (N_bars, 6): gmm2,km,vol,gs,cu,rq

    features, close_prices = prepare_features(df, window_size=args.window_size)
    ws = args.window_size
    features     = features[ws:]
    close_prices = close_prices[ws:]
    regime_arr   = regime_arr_full[ws:]

    # Align seq-level arrays without building the full X array (caller's change)
    sl = args.seq_length
    n_seq      = len(features) - sl
    seq_prices = close_prices[sl - 1 : sl - 1 + n_seq]
    seq_regime = regime_arr  [sl - 1 : sl - 1 + n_seq]
    seq_gmm2   = seq_regime[:, 0]
    seq_gs     = seq_regime[:, 3]
    seq_cu     = seq_regime[:, 4]
    seq_rq     = seq_regime[:, 5]

    logger.info(
        f"Sequences: {n_seq:,} | GMM2 Bull={seq_gmm2.mean():.1%} | "
        f"G/S Q1={np.mean(seq_gs<=0.25):.1%} Q4={np.mean(seq_gs>=0.75):.1%}"
    )

    # ── Frozen supervised model ───────────────────────────────────────────────
    ckpt = torch.load(str(ckpt_path), map_location=device)
    from omegaconf import OmegaConf
    cfg   = OmegaConf.create(ckpt.get("config", {}))
    model = DualBranchModel.from_config(cfg.model) if cfg else DualBranchModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(f"Frozen model: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Feature extraction via SequenceDataset streaming ─────────────────────
    signals, confidences = extract_model_features(
        model, features, sl, device, batch_size=args.batch_size
    )

    # ── v4 obs context features ───────────────────────────────────────────────
    try:
        raw_ts  = df["timestamp"].to_list() if "timestamp" in df.columns else None
        seq_ts  = raw_ts[ws + sl - 1 : ws + sl - 1 + len(signals)] if raw_ts else None
        if seq_ts and len(seq_ts) < len(signals):
            seq_ts = None
    except Exception:
        seq_ts = None

    atr_norm, trend_norm, session_phase = compute_rl_obs_features(seq_prices, seq_ts)

    split = int(len(signals) * 0.8)
    logger.info(f"Train={split:,} Eval={len(signals)-split:,}")

    def make_env(sl_idx, su_idx):
        return FrozenEncoderEnv(
            signals        = signals[sl_idx:su_idx],
            confidences    = confidences[sl_idx:su_idx],
            prices         = seq_prices[sl_idx:su_idx],
            atr_norm       = atr_norm[sl_idx:su_idx],
            trend_norm     = trend_norm[sl_idx:su_idx],
            session_phase  = session_phase[sl_idx:su_idx],
            regime_quality = seq_rq[sl_idx:su_idx],
            gs_quartile    = seq_gs[sl_idx:su_idx],
            cu_au_regime   = seq_cu[sl_idx:su_idx],
            gmm2_state     = seq_gmm2[sl_idx:su_idx],
            max_hold            = args.max_hold,
            episode_len         = args.episode_len,
            confidence_gate     = args.confidence_gate,
            buy_reward_scale    = args.buy_reward_scale,
            mtm_scale           = args.mtm_scale,
            hold_penalty_coeff  = args.hold_penalty,
            early_cut_bonus_frac= args.early_cut_bonus,
        )

    train_env = make_env(0, split)
    eval_env  = make_env(split, len(signals))

    agent = ConfidenceSACAgent(
        obs_dim=13,
        hidden_dims=[256, 256],
        device=str(device),
        curriculum_warmup_steps=args.curriculum_warmup,
    )
    logger.info(
        f"Phase 3 SAC agent v2: obs=13 | steps={args.steps:,}\n"
        f"  confidence_gate={args.confidence_gate} (was 0.48)\n"
        f"  buy_reward_scale={args.buy_reward_scale} (sell F1=0.357, buy F1=0.021)\n"
        f"  n_eval_episodes={args.n_eval_episodes} (was 3)\n"
        f"  G/S max_hold: Q1=40 Q2/3=60 Q4=80 | GMM2 Bear entry gate ON\n"
        f"  Curriculum warmup: {args.curriculum_warmup:,} steps"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    obs, _          = train_env.reset()
    best_eval_pnl   = -float("inf")
    ep_rewards, ep_pnls = [], []
    ep_reward = 0.0
    ep_count  = 0

    for step in range(1, args.steps + 1):
        agent.set_step(step)
        train_env.exit_threshold = agent._exit_threshold

        conf = float(obs[3])
        pos_action, should_exit = agent.select_action(obs, confidence=conf)
        action = np.array([pos_action, -1.0 if should_exit else 0.5], dtype=np.float32)

        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        agent.store(obs, action, reward, next_obs, done)
        ep_reward += reward
        obs = next_obs

        if agent.buffer_size >= agent.batch_size:
            agent.update()

        if done:
            ep_rewards.append(ep_reward)
            ep_pnls.append(info["episode_pnl"])
            ep_count  += 1
            ep_reward  = 0.0
            obs, _     = train_env.reset()

        if step % args.eval_every == 0:
            avg_r   = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
            avg_pnl = np.mean(ep_pnls[-20:])    if ep_pnls    else 0.0
            eval_pnl, eval_trades, eval_wr = evaluate(
                agent, eval_env, n_episodes=args.n_eval_episodes
            )
            logger.info(
                f"Step {step:,}/{args.steps:,} (ep={ep_count}) | "
                f"Train: r={avg_r:.2f} pnl=${avg_pnl:.2f} | "
                f"Eval({args.n_eval_episodes}ep): pnl=${eval_pnl:.2f} "
                f"trades={eval_trades:.0f} wr={eval_wr:.1%}"
            )
            if eval_pnl > best_eval_pnl:
                best_eval_pnl = eval_pnl
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                agent.save(f"{args.save_dir}/rl_agent_best.pt")
                logger.info(f"  → Best eval PnL: ${eval_pnl:.2f}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    final_pnl, final_trades, final_wr = evaluate(
        agent, eval_env, n_episodes=args.n_eval_episodes * 2
    )
    logger.info(
        f"\n{'='*60}\n"
        f"PHASE 3 RL v2 TRAINING COMPLETE\n"
        f"{'='*60}\n"
        f"Steps:          {args.steps:,}\n"
        f"Episodes:       {ep_count}\n"
        f"Best eval PnL:  ${best_eval_pnl:.2f}\n"
        f"Final eval PnL: ${final_pnl:.2f}\n"
        f"Final trades:   {final_trades:.0f}\n"
        f"Final win rate: {final_wr:.1%}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()