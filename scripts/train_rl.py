#!/usr/bin/env python3
"""Train RL agent on top of frozen dual-branch supervised model.

REQUIRES: models/dual_branch_best.pt (from supervised pre-training)

The frozen supervised model (52.9% acc, 2:1 R:R, conf ~0.916) provides
signal probabilities and confidence scores. The RL agent learns:
    1. WHEN to act   — skip low-confidence signals
    2. HOW MUCH      — confidence-based position sizing
    3. WHEN to exit  — flexible timing, not locked to label horizon

Observation (7-dim):
    [sell_prob, hold_prob, buy_prob, confidence, position_dir,
     unrealized_pnl_norm, hold_time_norm]

Action: continuous [-1, +1]
    -1 = max short, 0 = flat, +1 = max long
    |action| < 0.15 = go flat (close any position)

Reward: confidence-gated PnL
    reward = pnl_usd × max(confidence - gate, 0) / (1 - gate)
    Low-confidence trades → zero reward regardless of outcome.

Usage:
    python scripts/train_rl.py --checkpoint models/dual_branch_best.pt --steps 200000
    python scripts/train_rl.py --checkpoint models/dual_branch_best.pt --confidence-gate 0.35
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

from src.data.preprocessing import prepare_features
from src.data.tick_store import TickStore
from src.encoder.fusion import DualBranchModel
from src.meta_policy.rl_agent import ConfidenceSACAgent
from src.training.labels import create_sequences
from src.utils.config import set_seed
from src.utils.logger import setup_logger


# =========================================================================
# Environment
# =========================================================================

class FrozenEncoderEnv(gym.Env):
    """Trading env driven by pre-extracted supervised model outputs.

    Each episode samples a random contiguous window from the data,
    preventing the agent from memorizing a single trajectory.
    """

    FLAT_THRESHOLD = 0.15

    def __init__(
        self,
        signals: np.ndarray,       # (N, 3) softmax probabilities
        confidences: np.ndarray,    # (N,)   confidence [0,1]
        prices: np.ndarray,         # (N,)   close prices
        max_hold: int = 120,
        episode_len: int = 2000,    # bars per episode (random window)
        confidence_gate: float = 0.3,
        spread_pips: float = 2.0,
        pip_value: float = 0.10,
        commission_usd: float = 0.70,  # per 0.01 lot round-trip
        lot_size: float = 0.01,
        initial_balance: float = 10_000.0,
    ):
        super().__init__()
        assert len(signals) == len(confidences) == len(prices)

        self.signals = signals.astype(np.float32)
        self.confidences = confidences.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.max_hold = max_hold
        self.episode_len = min(episode_len, len(prices) - 1)
        self.confidence_gate = confidence_gate
        self.spread_cost = spread_pips * pip_value  # $0.20 total spread
        self.commission = commission_usd
        self.lot_size = lot_size
        self.initial_balance = initial_balance
        self.pip_value = pip_value

        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        self._offset = 0  # random start offset per episode
        self._reset_state()

    def _reset_state(self):
        self.local_idx = 0
        self.balance = self.initial_balance
        self.position_dir = 0       # -1, 0, +1
        self.entry_price = 0.0
        self.entry_conf = 0.0
        self.hold_time = 0
        self.n_trades = 0
        self.episode_pnl = 0.0
        self.n_wins = 0

    def _global_idx(self):
        return self._offset + self.local_idx

    def _current_price(self):
        return self.prices[self._global_idx()]

    def _get_obs(self) -> np.ndarray:
        gi = self._global_idx()
        sig = self.signals[gi]
        conf = self.confidences[gi]

        unreal = 0.0
        if self.position_dir != 0:
            price_diff = self._current_price() - self.entry_price
            # Normalize: $1 move on XAUUSD ≈ 10 pips → unrealized in pip-units / 100
            unreal = (price_diff * self.position_dir) / self.pip_value / 100.0

        return np.array([
            sig[0], sig[1], sig[2],
            conf,
            float(self.position_dir),
            np.clip(unreal, -5.0, 5.0),
            min(self.hold_time / max(self.max_hold, 1), 1.0),
        ], dtype=np.float32)

    def _close_position(self) -> float:
        """Close current position. Returns confidence-gated reward."""
        if self.position_dir == 0:
            return 0.0

        price = self._current_price()
        # Raw PnL in price units
        raw_pnl_price = (price - self.entry_price) * self.position_dir

        # Convert to USD: pnl_pips × $1/pip for 0.01 lot XAUUSD
        pnl_pips = raw_pnl_price / self.pip_value
        pnl_usd = pnl_pips * 1.0  # $1 per pip per 0.01 lot
        pnl_usd -= self.spread_cost / self.pip_value * 0.5  # half spread on close
        pnl_usd -= self.commission

        # Track actual balance
        self.balance += pnl_usd
        self.episode_pnl += pnl_usd
        self.n_trades += 1
        if pnl_usd > 0:
            self.n_wins += 1

        # Confidence gate: scale reward by entry confidence
        conf = self.entry_conf
        if conf <= self.confidence_gate:
            gated_reward = 0.0  # zero reward for low-confidence trades
        else:
            gate_scale = (conf - self.confidence_gate) / (1.0 - self.confidence_gate)
            gated_reward = pnl_usd * gate_scale

        # Reset position
        self.position_dir = 0
        self.entry_price = 0.0
        self.entry_conf = 0.0
        self.hold_time = 0

        return gated_reward

    def step(self, action):
        action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        reward = 0.0

        # Force close if max hold exceeded
        if self.position_dir != 0 and self.hold_time >= self.max_hold:
            reward += self._close_position()

        # Parse action
        if abs(action_val) < self.FLAT_THRESHOLD:
            # Go flat
            if self.position_dir != 0:
                reward += self._close_position()

        elif action_val > 0 and self.position_dir != 1:
            # Want long
            if self.position_dir == -1:
                reward += self._close_position()
            if self.position_dir == 0:
                self.position_dir = 1
                self.entry_price = self._current_price() + self.spread_cost * 0.5
                self.entry_conf = self.confidences[self._global_idx()]
                self.hold_time = 0

        elif action_val < 0 and self.position_dir != -1:
            # Want short
            if self.position_dir == 1:
                reward += self._close_position()
            if self.position_dir == 0:
                self.position_dir = -1
                self.entry_price = self._current_price() - self.spread_cost * 0.5
                self.entry_conf = self.confidences[self._global_idx()]
                self.hold_time = 0

        if self.position_dir != 0:
            self.hold_time += 1

        self.local_idx += 1
        terminated = self.local_idx >= self.episode_len
        truncated = self.balance <= 0

        if terminated or truncated:
            if self.position_dir != 0:
                reward += self._close_position()

        obs = self._get_obs() if not (terminated or truncated) else np.zeros(7, np.float32)
        info = {
            "balance": self.balance,
            "episode_pnl": self.episode_pnl,
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "win_rate": self.n_wins / max(self.n_trades, 1),
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        # Random start offset — different trajectory each episode
        max_offset = len(self.prices) - self.episode_len - 1
        if max_offset > 0:
            self._offset = np.random.randint(0, max_offset)
        else:
            self._offset = 0
        return self._get_obs(), {}


# =========================================================================
# Feature extraction
# =========================================================================

def extract_model_features(model, X, S, device, batch_size=512):
    """Run frozen supervised model on all sequences."""
    model.eval()
    all_probs, all_confs = [], []

    n_batches = (len(X) + batch_size - 1) // batch_size
    logger.info(f"Extracting features from {len(X)} sequences ({n_batches} batches)...")

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.FloatTensor(X[i:i+batch_size]).to(device)
            bs = torch.FloatTensor(S[i:i+batch_size]).to(device) if S is not None else None
            logits, conf = model(bx, bs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            confs = conf.squeeze(-1).cpu().numpy()
            all_probs.append(probs)
            all_confs.append(confs)

            if (i // batch_size + 1) % 100 == 0:
                logger.info(f"  batch {i // batch_size + 1}/{n_batches}")

    signals = np.concatenate(all_probs)
    confidences = np.concatenate(all_confs)

    # Stats
    argmax_labels = signals.argmax(axis=1)
    logger.info(
        f"Feature extraction complete:\n"
        f"  Signals: {signals.shape}\n"
        f"  Confidence: mean={confidences.mean():.3f} std={confidences.std():.3f} "
        f"min={confidences.min():.3f} max={confidences.max():.3f}\n"
        f"  Signal distribution: sell={np.mean(argmax_labels==0):.1%} "
        f"hold={np.mean(argmax_labels==1):.1%} buy={np.mean(argmax_labels==2):.1%}"
    )
    return signals, confidences


# =========================================================================
# Evaluation
# =========================================================================

def evaluate(agent, env, n_episodes=3):
    """Run evaluation episodes and average results."""
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, confidence=obs[3], eval_mode=True)
            obs, _, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated
        total_pnl += info["episode_pnl"]
        total_trades += info["n_trades"]
        total_wins += info["n_wins"]

    avg_pnl = total_pnl / n_episodes
    avg_trades = total_trades / n_episodes
    wr = total_wins / max(total_trades, 1)
    return avg_pnl, avg_trades, wr


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent on frozen dual-branch supervised model"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument(
        "--checkpoint", default="models/dual_branch_best.pt",
        help="Path to frozen supervised model checkpoint (REQUIRED)"
    )
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--max-hold", type=int, default=120,
                        help="Max bars before force-close")
    parser.add_argument("--episode-len", type=int, default=2000,
                        help="Bars per RL episode (random window)")
    parser.add_argument("--confidence-gate", type=float, default=0.3,
                        help="Min confidence for non-zero reward (0.0-1.0)")
    parser.add_argument("--seq-length", type=int, default=120,
                        help="Sequence length (must match supervised model)")
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-gan", action="store_true")
    args = parser.parse_args()

    setup_logger()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Validate checkpoint exists ──
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(
            f"FATAL: Supervised checkpoint not found at {ckpt_path}\n"
            f"The RL agent requires a pre-trained dual-branch model.\n"
            f"Run supervised training first:\n"
            f"  python scripts/train_supervised.py model=dual_branch data=xauusd\n"
            f"Then retry:\n"
            f"  python scripts/train_rl.py --checkpoint {ckpt_path}"
        )
        sys.exit(1)

    # ── Load data ──
    logger.info("Loading data...")
    store = TickStore(f"{args.data_dir}/ticks.duckdb")
    df = store.query_ohlcv(args.symbol, "M1")
    store.close()

    if df.is_empty():
        logger.error("No data. Run: python scripts/download_data.py --synthetic --days 90")
        sys.exit(1)

    features, close_prices = prepare_features(df, window_size=args.window_size)
    features = features[args.window_size:]
    close_prices = close_prices[args.window_size:]

    # Create sequences (labels are dummy — not needed for RL)
    dummy_labels = np.zeros(len(features), dtype=np.int64)
    X, _, _ = create_sequences(features, dummy_labels, args.seq_length, sentiment=None)
    seq_prices = close_prices[args.seq_length - 1 : args.seq_length - 1 + len(X)]
    logger.info(f"Data: {len(X):,} sequences, {len(seq_prices):,} prices")

    # ── Load & freeze supervised model ──
    logger.info(f"Loading frozen supervised model from {ckpt_path}...")
    checkpoint = torch.load(str(ckpt_path), map_location=device)

    from omegaconf import OmegaConf
    model_cfg = checkpoint.get("config", {})
    if model_cfg:
        cfg = OmegaConf.create(model_cfg)
        model = DualBranchModel.from_config(cfg.model)
        logger.info(f"Rebuilt model from checkpoint config")
    else:
        model = DualBranchModel()
        logger.warning("No config in checkpoint — using default DualBranchModel")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # FREEZE all weights — this is critical
    for param in model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Supervised model loaded and FROZEN: {n_params:,} params (0 trainable)")

    # ── Extract supervised features for entire dataset ──
    signals, confidences = extract_model_features(model, X, None, device)

    # ── Split: 80% train / 20% eval (chronological) ──
    split = int(len(signals) * 0.8)
    logger.info(f"Split: train={split:,} eval={len(signals)-split:,}")

    train_env = FrozenEncoderEnv(
        signals=signals[:split],
        confidences=confidences[:split],
        prices=seq_prices[:split],
        max_hold=args.max_hold,
        episode_len=args.episode_len,
        confidence_gate=args.confidence_gate,
    )
    eval_env = FrozenEncoderEnv(
        signals=signals[split:],
        confidences=confidences[split:],
        prices=seq_prices[split:],
        max_hold=args.max_hold,
        episode_len=args.episode_len,
        confidence_gate=args.confidence_gate,
    )

    logger.info(
        f"RL Environment:\n"
        f"  obs_dim=7, action=continuous[-1,+1]\n"
        f"  max_hold={args.max_hold}, episode_len={args.episode_len}\n"
        f"  confidence_gate={args.confidence_gate}\n"
        f"  spread_cost=${train_env.spread_cost:.2f}, commission=${train_env.commission:.2f}"
    )

    # ── Create RL agent ──
    agent = ConfidenceSACAgent(obs_dim=7, hidden_dims=[256, 256], device=str(device))
    logger.info(f"SAC agent created: obs=7, hidden=[256,256]")

    # ── Training loop ──
    logger.info(f"Starting RL training: {args.steps:,} steps, eval every {args.eval_every:,}")

    obs, _ = train_env.reset()
    best_eval_pnl = -float("inf")
    episode_rewards = []
    episode_pnls = []
    ep_reward = 0
    ep_count = 0

    for step in range(1, args.steps + 1):
        # Select action using confidence from observation[3]
        conf = float(obs[3])
        action = agent.select_action(obs, confidence=conf)

        next_obs, reward, terminated, truncated, info = train_env.step(np.array([action]))
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)
        ep_reward += reward
        obs = next_obs

        # Update agent
        if agent.buffer_size >= agent.batch_size:
            agent.update()

        if done:
            episode_rewards.append(ep_reward)
            episode_pnls.append(info["episode_pnl"])
            ep_count += 1
            ep_reward = 0
            obs, _ = train_env.reset()

        # Periodic evaluation
        if step % args.eval_every == 0:
            avg_r = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            avg_pnl = np.mean(episode_pnls[-20:]) if episode_pnls else 0

            eval_pnl, eval_trades, eval_wr = evaluate(agent, eval_env, n_episodes=3)

            logger.info(
                f"Step {step:,}/{args.steps:,} (ep={ep_count}) | "
                f"Train: reward={avg_r:.2f} pnl=${avg_pnl:.2f} | "
                f"Eval: pnl=${eval_pnl:.2f} trades={eval_trades:.0f} wr={eval_wr:.1%}"
            )

            if eval_pnl > best_eval_pnl:
                best_eval_pnl = eval_pnl
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                agent.save(f"{args.save_dir}/rl_agent_best.pt")
                logger.info(f"  → New best eval PnL: ${eval_pnl:.2f}")

    # ── Final evaluation ──
    final_pnl, final_trades, final_wr = evaluate(agent, eval_env, n_episodes=5)
    logger.info(
        f"\n{'='*60}\n"
        f"RL TRAINING COMPLETE\n"
        f"{'='*60}\n"
        f"Total steps:     {args.steps:,}\n"
        f"Total episodes:  {ep_count}\n"
        f"Best eval PnL:   ${best_eval_pnl:.2f}\n"
        f"Final eval PnL:  ${final_pnl:.2f}\n"
        f"Final trades:    {final_trades:.0f}\n"
        f"Final win rate:  {final_wr:.1%}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
