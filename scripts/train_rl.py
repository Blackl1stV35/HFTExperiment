#!/usr/bin/env python3
"""Train RL agent on frozen dual-branch supervised model.

PHASE 3 changes (Steps 4, 5, 6):

  Step 4 — obs 10→13 dims:
    Adds regime_quality_norm, gs_quartile_norm, cu_au_regime_enc from the
    daily_regime_labels.csv (joined to M1 bars by date). The agent now sees:
      [sell, hold, buy, conf, pos_dir, unreal, hold_t,        ← 7 (unchanged)
       atr_norm, trend_norm, session_phase,                   ← 3 (PATCH v4)
       regime_quality_norm, gs_quartile_norm, cu_au_regime]   ← 3 (Phase 3 NEW)

  Step 5 — G/S-conditioned max_hold:
    max_hold is dynamically set per-bar based on the G/S quartile signal:
      Q1 (silver-leads, gs_q ≤ 0.25): 40 bars — commodity/mean-reverting
      Q2/Q3               (0.25–0.75): 60 bars — neutral
      Q4 (gold-leads,    gs_q > 0.75): 80 bars — fear-bid/trending
    This converts the research finding (7-day Q1 optimal daily hold) into an
    M1-scale approximation of hold character without redefining the training
    episode horizon.

  Step 6 — entry gate:
    New entries only allowed when gmm2_state == 1.0 (Bull).
    Bear regime (gmm2_state == 0.0) blocks new positions but does not force
    close existing ones (avoids mid-trade regime noise).

Usage:
    python scripts/train_rl.py \\
        --checkpoint models/dual_branch_best.pt \\
        --steps 500000 --seed 42 \\
        --eval-every 8000 \\
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

import gymnasium as gym
from gymnasium import spaces

from src.data.preprocessing import prepare_features, join_regime_labels, get_regime_array
from src.data.tick_store import TickStore
from src.data.feature_engineering import compute_rl_obs_features
from src.encoder.fusion import DualBranchModel
from src.meta_policy.rl_agent import ConfidenceSACAgent
from src.training.labels import create_sequences
from src.utils.config import set_seed
from src.utils.logger import setup_logger


# ─────────────────────────────────────────────────────────────────────────────
# Environment — Phase 3
# ─────────────────────────────────────────────────────────────────────────────

class FrozenEncoderEnv(gym.Env):
    """Trading env driven by frozen supervised model outputs.

    Phase 3 changes vs PATCH v4:
        - obs_dim 10 → 13 (+ regime_quality_norm, gs_quartile_norm, cu_au_regime)
        - max_hold per bar determined by G/S quartile (Step 5)
        - New entries blocked in GMM2 Bear regime (Step 6)
    """

    # G/S-conditioned hold limits (Step 5)
    _GS_HOLD = {
        "Q1": 40,   # silver-leads: commodity/mean-reverting → short hold
        "Q23": 60,  # neutral
        "Q4": 80,   # gold-leads: fear-bid/trending → full hold
    }

    def __init__(
        self,
        signals: np.ndarray,          # (N, 3)
        confidences: np.ndarray,       # (N,)
        prices: np.ndarray,            # (N,)
        atr_norm: np.ndarray | None = None,
        trend_norm: np.ndarray | None = None,
        session_phase: np.ndarray | None = None,
        # Phase 3: regime arrays (N,) each, float32
        regime_quality: np.ndarray | None = None,   # [0,1] heatmap Sharpe
        gs_quartile: np.ndarray | None = None,      # [0,1] Q1=0 Q4=1
        cu_au_regime: np.ndarray | None = None,     # [0,1] Financial/Mixed/Commodity
        gmm2_state: np.ndarray | None = None,       # 0=Bear 1=Bull
        max_hold: int = 80,
        episode_len: int = 2000,
        confidence_gate: float = 0.48,
        spread_pips: float = 2.0,
        pip_value: float = 0.10,
        commission_usd: float = 0.70,
        lot_size: float = 0.01,
        initial_balance: float = 10_000.0,
        mtm_scale: float = 0.05,
        hold_penalty_coeff: float = 0.003,
        early_cut_bonus_frac: float = 0.40,
    ):
        super().__init__()
        n = len(prices)
        assert len(signals) == len(confidences) == n

        self.signals     = signals.astype(np.float32)
        self.confidences = confidences.astype(np.float32)
        self.prices      = prices.astype(np.float64)

        # v4 obs arrays
        self.atr_norm      = atr_norm.astype(np.float32)      if atr_norm      is not None else np.zeros(n, np.float32)
        self.trend_norm    = trend_norm.astype(np.float32)     if trend_norm    is not None else np.zeros(n, np.float32)
        self.session_phase = session_phase.astype(np.float32)  if session_phase is not None else np.full(n, 0.5, np.float32)

        # Phase 3 regime arrays (Step 4)
        self.regime_quality = regime_quality.astype(np.float32) if regime_quality is not None else np.full(n, 0.5, np.float32)
        self.gs_quartile    = gs_quartile.astype(np.float32)    if gs_quartile    is not None else np.zeros(n, np.float32)
        self.cu_au_regime   = cu_au_regime.astype(np.float32)   if cu_au_regime   is not None else np.full(n, 0.5, np.float32)
        self.gmm2_state     = gmm2_state.astype(np.float32)     if gmm2_state     is not None else np.ones(n, np.float32)  # default Bull

        self._max_hold_default = max_hold
        self.episode_len   = min(episode_len, n - 1)
        self.confidence_gate = confidence_gate
        self.spread_cost   = spread_pips * pip_value
        self.commission    = commission_usd
        self.lot_size      = lot_size
        self.initial_balance = initial_balance
        self.pip_value     = pip_value
        self.mtm_scale     = mtm_scale
        self.hold_penalty_coeff   = hold_penalty_coeff
        self.early_cut_bonus_frac = early_cut_bonus_frac
        self.exit_threshold: float = 0.0

        # Phase 3: obs 13-dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (13,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self._offset = 0
        self._reset_state()

    def _reset_state(self):
        self.local_idx = 0
        self.balance = self.initial_balance
        self.position_dir = 0
        self.entry_price = 0.0
        self.entry_conf = 0.0
        self.hold_time = 0
        self.n_trades = 0
        self.episode_pnl = 0.0
        self.n_wins = 0
        self._prev_unrealized = 0.0
        self.max_hold = self._max_hold_default  # will be overridden per-bar

    def _gi(self):
        return self._offset + self.local_idx

    def _price(self):
        return self.prices[self._gi()]

    def _gs_max_hold(self) -> int:
        """Step 5: G/S-conditioned max hold."""
        gs = float(self.gs_quartile[self._gi()])
        if gs <= 0.25:
            return self._GS_HOLD["Q1"]   # 40 bars
        elif gs >= 0.75:
            return self._GS_HOLD["Q4"]   # 80 bars
        return self._GS_HOLD["Q23"]       # 60 bars

    def _unrealized_usd(self) -> float:
        if self.position_dir == 0:
            return 0.0
        return ((self._price() - self.entry_price) * self.position_dir / self.pip_value)

    def _get_obs(self) -> np.ndarray:
        gi = self._gi()
        sig   = self.signals[gi]
        conf  = self.confidences[gi]
        unreal = np.clip(self._unrealized_usd() / 100.0, -5.0, 5.0) if self.position_dir else 0.0
        hold_frac = min(self.hold_time / max(self.max_hold, 1), 1.0)
        return np.array([
            sig[0], sig[1], sig[2],          # sell/hold/buy probs
            conf,                             # model confidence
            float(self.position_dir),        # -1/0/1
            unreal,                           # unrealized PnL norm
            hold_frac,                        # hold time fraction
            self.atr_norm[gi],               # volatility
            self.trend_norm[gi],             # trend slope
            self.session_phase[gi],          # session time
            self.regime_quality[gi],         # Step 4: heatmap Sharpe [0,1]
            self.gs_quartile[gi],            # Step 4: G/S quartile [0,1]
            self.cu_au_regime[gi],           # Step 4: Cu-Au regime [0,1]
        ], dtype=np.float32)

    def _close_position(self, voluntary: bool = False) -> float:
        if self.position_dir == 0:
            return 0.0
        raw_pnl = (self._price() - self.entry_price) * self.position_dir / self.pip_value
        pnl_usd = raw_pnl - self.spread_cost / self.pip_value * 0.5 - self.commission
        self.balance += pnl_usd
        self.episode_pnl += pnl_usd
        self.n_trades += 1
        if pnl_usd > 0:
            self.n_wins += 1

        early_cut_bonus = 0.0
        if voluntary and pnl_usd < 0 and self.hold_time < self.max_hold * 0.8:
            remaining = self.max_hold - self.hold_time
            avoided = (abs(pnl_usd) / max(self.hold_time, 1)) * remaining
            early_cut_bonus = min(avoided * self.early_cut_bonus_frac, abs(pnl_usd) * 0.5)

        conf = self.entry_conf
        if conf <= self.confidence_gate:
            gated = 0.0
        else:
            gate_scale = (conf - self.confidence_gate) / (1.0 - self.confidence_gate)
            gated = pnl_usd * gate_scale

        gated += early_cut_bonus
        self.position_dir = 0
        self.entry_price = 0.0
        self.entry_conf = 0.0
        self.hold_time = 0
        self._prev_unrealized = 0.0
        return gated

    def step(self, action: np.ndarray):
        gi = self._gi()
        pos_action = float(action[0])
        exit_logit = float(action[1]) if len(action) > 1 else 0.0
        reward = 0.0
        should_exit = exit_logit < self.exit_threshold

        # Step 5: update max_hold from G/S quartile each bar
        self.max_hold = self._gs_max_hold()

        # Force-close at max_hold
        if self.position_dir != 0 and self.hold_time >= self.max_hold:
            reward += self._close_position(voluntary=False)

        # Voluntary exit
        if should_exit and self.position_dir != 0:
            reward += self._close_position(voluntary=True)

        # Step 6: entry gate — block new positions in Bear regime
        in_bull = float(self.gmm2_state[gi]) > 0.5

        if not should_exit:
            if pos_action > 0.0 and self.position_dir != 1:
                if self.position_dir == -1:
                    reward += self._close_position(voluntary=True)
                # Only open long in Bull regime
                if self.position_dir == 0 and in_bull:
                    self.position_dir = 1
                    self.entry_price = self._price() + self.spread_cost * 0.5
                    self.entry_conf = self.confidences[gi]
                    self.hold_time = 0
                    self._prev_unrealized = 0.0

            elif pos_action < 0.0 and self.position_dir != -1:
                if self.position_dir == 1:
                    reward += self._close_position(voluntary=True)
                # Only open short in Bull regime (short-selling against the trend)
                if self.position_dir == 0 and in_bull:
                    self.position_dir = -1
                    self.entry_price = self._price() - self.spread_cost * 0.5
                    self.entry_conf = self.confidences[gi]
                    self.hold_time = 0
                    self._prev_unrealized = 0.0

        # Per-step reward shaping
        if self.position_dir != 0:
            self.hold_time += 1
            cur_unreal = self._unrealized_usd()
            reward += (cur_unreal - self._prev_unrealized) * self.mtm_scale
            self._prev_unrealized = cur_unreal
            hold_frac = self.hold_time / max(self.max_hold, 1)
            reward -= self.hold_penalty_coeff * (hold_frac ** 2)

        self.local_idx += 1
        terminated = self.local_idx >= self.episode_len
        truncated  = self.balance <= 0

        if terminated or truncated:
            if self.position_dir != 0:
                reward += self._close_position(voluntary=False)

        obs  = self._get_obs() if not (terminated or truncated) else np.zeros(13, np.float32)
        info = {"balance": self.balance, "episode_pnl": self.episode_pnl,
                "n_trades": self.n_trades, "n_wins": self.n_wins,
                "win_rate": self.n_wins / max(self.n_trades, 1),
                "max_hold": self.max_hold}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        max_offset = len(self.prices) - self.episode_len - 1
        self._offset = np.random.randint(0, max_offset) if max_offset > 0 else 0
        return self._get_obs(), {}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    """Sliding-window dataset that avoids pre-allocating the full X array."""
    def __init__(self, features: np.ndarray, seq_len: int):
        self.features = features
        self.seq_len  = seq_len
        self._n       = len(features) - seq_len

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.features[i : i + self.seq_len].copy())
        return x

def extract_model_features(model, features, seq_len, device, batch_size=2048):
    """Memory-efficient feature extraction using a sliding window DataLoader."""
    model.eval()
    
    # Use the lightweight sliding-window dataset
    dataset = SequenceDataset(features, seq_len)
    loader  = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    all_probs, all_confs = [], []
    
    logger.info("Extracting signals batch-by-batch (this takes ~2 mins)...")
    with torch.no_grad():
        for bx in loader:
            bx = bx.to(device)
            # Pass None for sentiment (S) since we disabled it
            logits, conf = model(bx, None) 
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_confs.append(conf.squeeze(-1).cpu().numpy())
            
    signals     = np.concatenate(all_probs)
    confidences = np.concatenate(all_confs)
    argmax      = signals.argmax(axis=1)
    
    logger.info(
        f"Features: {signals.shape} | conf {confidences.mean():.3f}±{confidences.std():.3f} | "
        f"sell={np.mean(argmax==0):.1%} hold={np.mean(argmax==1):.1%} buy={np.mean(argmax==2):.1%}"
    )
    return signals, confidences


def evaluate(agent, env, n_episodes=3):
    total_pnl = 0.0; total_trades = 0; total_wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            pos_action, should_exit = agent.select_action(obs, confidence=obs[3], eval_mode=True)
            action = np.array([pos_action, -1.0 if should_exit else 0.5], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        total_pnl    += info["episode_pnl"]
        total_trades += info["n_trades"]
        total_wins   += info["n_wins"]
    return total_pnl/n_episodes, total_trades/n_episodes, total_wins/max(total_trades, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3 RL training (obs=13, G/S hold, GMM2 gate)")
    parser.add_argument("--data-dir",           default="data")
    parser.add_argument("--symbol",             default="XAUUSD")
    parser.add_argument("--checkpoint",         default="models/dual_branch_best.pt")
    parser.add_argument("--steps",   type=int,  default=500_000)
    parser.add_argument("--max-hold",type=int,  default=80,      help="Default; overridden by G/S quartile per-bar")
    parser.add_argument("--episode-len",type=int,default=2000)
    parser.add_argument("--confidence-gate",type=float,default=0.48)
    parser.add_argument("--seq-length",  type=int,  default=120)
    parser.add_argument("--window-size", type=int,  default=120)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--save-dir",    default="models")
    parser.add_argument("--eval-every",  type=int,  default=8_000)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--mtm-scale",         type=float, default=0.05)
    parser.add_argument("--hold-penalty",      type=float, default=0.003)
    parser.add_argument("--early-cut-bonus",   type=float, default=0.40)
    parser.add_argument("--curriculum-warmup", type=int,   default=100_000)
    parser.add_argument("--regime-csv",
        default="data/regime/daily_regime_labels.csv",
        help="Path to daily_regime_labels.csv from research notebook")
    args = parser.parse_args()

    setup_logger()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    logger.info(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run supervised training first:\n"
            f"  python scripts/train_supervised.py model=dual_branch data=xauusd"
        )
        sys.exit(1)

    # Load and join regime labels
    store = TickStore(f"{args.data_dir}/ticks.duckdb")
    df    = store.query_ohlcv(args.symbol, "M1")
    store.close()
    if df.is_empty():
        logger.error("No data. Run: python scripts/download_data.py"); sys.exit(1)

    df = join_regime_labels(df, args.regime_csv)

    # Extract regime arrays (will be aligned to sequence count below)
    regime_arr_full = get_regime_array(df)   # (N_bars, 6)
    # cols: [gmm2, km_63d, vol, gs_q, cu_au, rq]

    features, close_prices = prepare_features(df, window_size=args.window_size)
    ws = args.window_size
    features    = features[ws:]
    close_prices = close_prices[ws:]
    regime_arr  = regime_arr_full[ws:]

    # 1. Calculate how many sequences we WILL have
    n_seqs = len(features) - args.seq_length

    # 2. Slice the trailing arrays directly (no more create_sequences!)
    seq_prices = close_prices[args.seq_length - 1 : args.seq_length - 1 + n_seqs]
    seq_regime = regime_arr  [args.seq_length - 1 : args.seq_length - 1 + n_seqs]
    
    # regime cols: gmm2=0, km=1, vol=2, gs=3, cu=4, rq=5
    seq_gmm2   = seq_regime[:, 0]
    seq_gs     = seq_regime[:, 3]
    seq_cu     = seq_regime[:, 4]
    seq_rq     = seq_regime[:, 5]

    logger.info(f"Sequences: {n_seqs:,} | GMM2 Bull={seq_gmm2.mean():.1%} | "
                f"G/S Q1={np.mean(seq_gs<=0.25):.1%} Q4={np.mean(seq_gs>=0.75):.1%}")

    # Load frozen model
    ckpt = torch.load(str(ckpt_path), map_location=device)
    from omegaconf import OmegaConf
    cfg  = OmegaConf.create(ckpt.get("config", {}))
    model = DualBranchModel.from_config(cfg.model) if cfg else DualBranchModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad = False
    logger.info(f"Frozen model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # 3. Stream the raw features through the model efficiently
    signals, confidences = extract_model_features(model, features, args.seq_length, device)

    # Compute v4 obs features
    try:
        df_ts = TickStore(f"{args.data_dir}/ticks.duckdb").query_ohlcv(args.symbol, "M1")
        raw_ts = df_ts["timestamp"].to_list()
        offset = ws + args.seq_length - 1
        seq_ts = raw_ts[offset : offset + len(signals)]
        if len(seq_ts) < len(signals): seq_ts = None
    except Exception:
        seq_ts = None

    atr_norm, trend_norm, session_phase = compute_rl_obs_features(seq_prices, seq_ts)

    split = int(len(signals) * 0.8)
    logger.info(f"Train={split:,} Eval={len(signals)-split:,}")

    def make_env(sl, su):
        return FrozenEncoderEnv(
            signals=signals[sl:su], confidences=confidences[sl:su],
            prices=seq_prices[sl:su],
            atr_norm=atr_norm[sl:su], trend_norm=trend_norm[sl:su],
            session_phase=session_phase[sl:su],
            regime_quality=seq_rq[sl:su],
            gs_quartile=seq_gs[sl:su],
            cu_au_regime=seq_cu[sl:su],
            gmm2_state=seq_gmm2[sl:su],
            max_hold=args.max_hold,
            episode_len=args.episode_len,
            confidence_gate=args.confidence_gate,
            mtm_scale=args.mtm_scale,
            hold_penalty_coeff=args.hold_penalty,
            early_cut_bonus_frac=args.early_cut_bonus,
        )

    train_env = make_env(0, split)
    eval_env  = make_env(split, len(signals))

    # Step 6: obs_dim=13
    agent = ConfidenceSACAgent(
        obs_dim=13,
        hidden_dims=[256, 256],
        device=str(device),
        curriculum_warmup_steps=args.curriculum_warmup,
    )
    logger.info(
        f"Phase 3 SAC agent: obs=13 (+regime_quality, gs_quartile, cu_au_regime)\n"
        f"  G/S-conditioned max_hold: Q1→40 Q2/3→60 Q4→80 bars\n"
        f"  GMM2 Bear entry gate: new positions blocked when gmm2_state=0\n"
        f"  Curriculum warmup: {args.curriculum_warmup:,} steps"
    )

    obs, _ = train_env.reset()
    best_eval_pnl = -float("inf")
    ep_rewards, ep_pnls = [], []
    ep_reward = 0.0; ep_count = 0

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
            ep_rewards.append(ep_reward); ep_pnls.append(info["episode_pnl"])
            ep_count += 1; ep_reward = 0.0
            obs, _ = train_env.reset()

        if step % args.eval_every == 0:
            avg_r   = np.mean(ep_rewards[-20:]) if ep_rewards else 0
            avg_pnl = np.mean(ep_pnls[-20:])    if ep_pnls    else 0
            eval_pnl, eval_trades, eval_wr = evaluate(agent, eval_env, n_episodes=3)
            logger.info(
                f"Step {step:,}/{args.steps:,} (ep={ep_count}) | "
                f"Train: r={avg_r:.2f} pnl=${avg_pnl:.2f} | "
                f"Eval: pnl=${eval_pnl:.2f} trades={eval_trades:.0f} wr={eval_wr:.1%}"
            )
            if eval_pnl > best_eval_pnl:
                best_eval_pnl = eval_pnl
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                agent.save(f"{args.save_dir}/rl_agent_best.pt")
                logger.info(f"  → Best eval PnL: ${eval_pnl:.2f}")

    final_pnl, final_trades, final_wr = evaluate(agent, eval_env, n_episodes=5)
    logger.info(
        f"\n{'='*60}\n"
        f"PHASE 3 RL TRAINING COMPLETE\n"
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
