#!/usr/bin/env python3
"""Train RL agent on top of frozen dual-branch supervised model.

The supervised model (52.9% accuracy, 2:1 R:R) is used as a fixed feature
extractor. The RL agent learns:
    1. WHEN to act (skip low-confidence signals)
    2. HOW MUCH to bet (confidence-based position sizing)
    3. WHEN to exit (flexible exit timing, not locked to 60-bar label horizon)

Architecture:
    Frozen DualBranch → [logits(3), confidence(1)] → RL observation
    RL observation = [sell_prob, hold_prob, buy_prob, confidence, position_dir,
                      unrealized_pnl_normalized, hold_time_normalized]
    RL action = continuous [-1, +1] (short ← 0 → long), modulated by confidence

Reward:
    reward = realized_pnl × max(confidence - 0.3, 0) / 0.7
    This makes low-confidence trades worth zero reward regardless of outcome.

Usage:
    python scripts/train_rl.py --checkpoint models/dual_branch_best.pt
    python scripts/train_rl.py --checkpoint models/dual_branch_best.pt --use-gan
    python scripts/train_rl.py --checkpoint models/dual_branch_best.pt --steps 1000000
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
from src.meta_policy.gan_market import GANMarketSimulator
from src.training.labels import create_sequences
from src.utils.config import set_seed, setup_logger


class FrozenEncoderEnv(gym.Env):
    """Trading environment that uses frozen supervised model as feature extractor.

    Observation space (7 dims):
        [0] sell_probability    — from softmax(logits)
        [1] hold_probability
        [2] buy_probability
        [3] confidence          — from confidence head [0, 1]
        [4] position_direction  — -1 (short), 0 (flat), +1 (long)
        [5] unrealized_pnl_norm — normalized by entry price
        [6] hold_time_norm      — normalized by max_hold

    Action space: continuous [-1, +1]
        -1.0 = maximum short conviction
         0.0 = flat / no position
        +1.0 = maximum long conviction
        |action| < 0.15 = close any existing position and go flat

    Reward:
        Confidence-gated PnL: reward = pnl × max(confidence - 0.3, 0) / 0.7
    """

    FLAT_THRESHOLD = 0.15
    SPREAD_COST = 2.0 * 0.10
    COMMISSION = 7.0

    def __init__(
        self,
        model_signals: np.ndarray,
        model_confidences: np.ndarray,
        prices: np.ndarray,
        max_hold: int = 120,
        initial_balance: float = 10_000.0,
        confidence_gate: float = 0.3,
        lot_size: float = 0.01,
    ):
        super().__init__()
        self.signals = model_signals.astype(np.float32)
        self.confidences = model_confidences.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.max_hold = max_hold
        self.initial_balance = initial_balance
        self.confidence_gate = confidence_gate
        self.lot_size = lot_size

        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self._reset_state()

    def _reset_state(self):
        self.step_idx = 0
        self.balance = self.initial_balance
        self.position_dir = 0
        self.entry_price = 0.0
        self.entry_confidence = 0.0
        self.hold_time = 0
        self.n_trades = 0
        self.episode_pnl = 0.0

    def _get_obs(self) -> np.ndarray:
        sig = self.signals[self.step_idx]
        conf = self.confidences[self.step_idx]
        unreal_pnl = 0.0
        if self.position_dir != 0 and self.entry_price > 0:
            unreal_pnl = (self.prices[self.step_idx] - self.entry_price) * self.position_dir
            unreal_pnl /= (self.entry_price * 0.01)
        return np.array([
            sig[0], sig[1], sig[2], conf,
            float(self.position_dir), unreal_pnl,
            self.hold_time / self.max_hold,
        ], dtype=np.float32)

    def _execute_close(self) -> float:
        if self.position_dir == 0:
            return 0.0
        price = self.prices[self.step_idx]
        raw_pnl = (price - self.entry_price) * self.position_dir
        spread = self.SPREAD_COST
        commission = self.COMMISSION * (self.lot_size / 0.01)
        pnl = raw_pnl * (self.lot_size / 0.01) * 10.0 - spread - commission

        conf = self.entry_confidence
        gate = max(conf - self.confidence_gate, 0.0) / (1.0 - self.confidence_gate)
        gated_pnl = pnl * gate

        self.position_dir = 0
        self.entry_price = 0.0
        self.entry_confidence = 0.0
        self.hold_time = 0
        self.n_trades += 1
        self.balance += pnl
        self.episode_pnl += pnl
        return gated_pnl

    def step(self, action):
        action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        reward = 0.0

        if self.position_dir != 0 and self.hold_time >= self.max_hold:
            reward += self._execute_close()

        if abs(action_val) < self.FLAT_THRESHOLD:
            if self.position_dir != 0:
                reward += self._execute_close()
        elif action_val > 0 and self.position_dir != 1:
            if self.position_dir == -1:
                reward += self._execute_close()
            if self.position_dir == 0:
                self.position_dir = 1
                self.entry_price = self.prices[self.step_idx] + self.SPREAD_COST * 0.5
                self.entry_confidence = self.confidences[self.step_idx]
                self.hold_time = 0
        elif action_val < 0 and self.position_dir != -1:
            if self.position_dir == 1:
                reward += self._execute_close()
            if self.position_dir == 0:
                self.position_dir = -1
                self.entry_price = self.prices[self.step_idx] - self.SPREAD_COST * 0.5
                self.entry_confidence = self.confidences[self.step_idx]
                self.hold_time = 0

        if self.position_dir != 0:
            self.hold_time += 1

        self.step_idx += 1
        terminated = self.step_idx >= len(self.prices) - 1
        truncated = self.balance <= 0

        if terminated or truncated:
            if self.position_dir != 0:
                reward += self._execute_close()

        obs = self._get_obs() if not (terminated or truncated) else np.zeros(7, dtype=np.float32)
        info = {"balance": self.balance, "episode_pnl": self.episode_pnl, "n_trades": self.n_trades, "position": self.position_dir}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}


def extract_model_features(model, X, S, device, batch_size=512):
    """Run frozen supervised model on all data."""
    model.eval()
    all_probs, all_confs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.FloatTensor(X[i:i+batch_size]).to(device)
            bs = torch.FloatTensor(S[i:i+batch_size]).to(device) if S is not None else None
            logits, conf = model(bx, bs)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_confs.append(conf.squeeze(-1).cpu().numpy())
    signals = np.concatenate(all_probs)
    confidences = np.concatenate(all_confs)
    logger.info(f"Extracted: signals={signals.shape} mean_conf={confidences.mean():.3f} "
                f"sell={np.mean(signals.argmax(1)==0):.1%} hold={np.mean(signals.argmax(1)==1):.1%} buy={np.mean(signals.argmax(1)==2):.1%}")
    return signals, confidences


def evaluate(agent, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs, confidence=obs[3], eval_mode=True)
        obs, _, terminated, truncated, info = env.step(np.array([action]))
        done = terminated or truncated
    trades = info["n_trades"]
    wr = max(0, min(1, (info["balance"] - 10_000) / (abs(info["episode_pnl"]) + 1e-8) * 0.5 + 0.5)) if trades > 0 else 0.5
    return info["episode_pnl"], trades, wr


def main():
    parser = argparse.ArgumentParser(description="Train RL on frozen supervised model")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--checkpoint", default="models/dual_branch_best.pt")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--max-hold", type=int, default=120)
    parser.add_argument("--confidence-gate", type=float, default=0.3)
    parser.add_argument("--use-gan", action="store_true")
    parser.add_argument("--seq-length", type=int, default=120)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    setup_logger()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    # Load data
    store = TickStore(f"{args.data_dir}/ticks.duckdb")
    df = store.query_ohlcv(args.symbol, "M1")
    store.close()
    if df.is_empty():
        logger.error("No data.")
        return

    features, close_prices = prepare_features(df, window_size=args.window_size)
    features = features[args.window_size:]
    close_prices = close_prices[args.window_size:]

    dummy_labels = np.zeros(len(features), dtype=np.int64)
    X, _, S = create_sequences(features, dummy_labels, args.seq_length, sentiment=None)
    seq_prices = close_prices[args.seq_length - 1 : args.seq_length - 1 + len(X)]
    logger.info(f"Data: {len(X)} sequences, {len(seq_prices)} prices")

    # Load frozen model
    ckpt_path = args.checkpoint
    if not Path(ckpt_path).exists():
        logger.error(f"Checkpoint not found: {ckpt_path}. Run make train first.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    from omegaconf import OmegaConf
    model_cfg = checkpoint.get("config", {})
    if model_cfg:
        cfg = OmegaConf.create(model_cfg)
        model = DualBranchModel.from_config(cfg.model)
    else:
        model = DualBranchModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(f"Frozen supervised model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Extract features
    signals, confidences = extract_model_features(model, X, S, device)

    # Split 80/20
    split = int(len(signals) * 0.8)
    train_env = FrozenEncoderEnv(signals[:split], confidences[:split], seq_prices[:split],
                                  max_hold=args.max_hold, confidence_gate=args.confidence_gate)
    eval_env = FrozenEncoderEnv(signals[split:], confidences[split:], seq_prices[split:],
                                 max_hold=args.max_hold, confidence_gate=args.confidence_gate)

    # Optional GAN
    if args.use_gan:
        logger.info("Training GAN market simulator...")
        from torch.utils.data import DataLoader, TensorDataset
        gan = GANMarketSimulator(seq_len=args.seq_length, feature_dim=6, device=str(device))
        real_seqs = np.array([features[i:i+args.seq_length] for i in range(0, len(features)-args.seq_length, args.seq_length)])
        loader = DataLoader(TensorDataset(torch.FloatTensor(real_seqs)), batch_size=32, shuffle=True)
        for epoch in range(50):
            for (batch,) in loader:
                gan.train_step(batch.to(device))
            if (epoch+1) % 10 == 0:
                logger.info(f"GAN epoch {epoch+1}/50")
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        gan.save(f"{args.save_dir}/gan_market.pt")

    # RL agent
    agent = ConfidenceSACAgent(obs_dim=7, hidden_dims=[256, 256], device=str(device))

    # Training
    obs, _ = train_env.reset()
    best_eval_pnl = -float("inf")
    episode_rewards, episode_pnls = [], []
    ep_reward = 0

    for step in range(1, args.steps + 1):
        action = agent.select_action(obs, confidence=obs[3])
        next_obs, reward, terminated, truncated, info = train_env.step(np.array([action]))
        done = terminated or truncated
        agent.store(obs, action, reward, next_obs, done)
        ep_reward += reward
        obs = next_obs

        if agent.buffer_size >= agent.batch_size:
            agent.update()

        if done:
            episode_rewards.append(ep_reward)
            episode_pnls.append(info["episode_pnl"])
            ep_reward = 0
            obs, _ = train_env.reset()

        if step % args.eval_every == 0:
            avg_r = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            avg_pnl = np.mean(episode_pnls[-20:]) if episode_pnls else 0
            eval_pnl, eval_trades, eval_wr = evaluate(agent, eval_env)
            logger.info(f"Step {step:,}/{args.steps:,} | Train: reward={avg_r:.2f} pnl=${avg_pnl:.2f} | "
                        f"Eval: pnl=${eval_pnl:.2f} trades={eval_trades} wr={eval_wr:.1%}")
            if eval_pnl > best_eval_pnl:
                best_eval_pnl = eval_pnl
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                agent.save(f"{args.save_dir}/rl_agent_best.pt")
                logger.info(f"  → New best: ${eval_pnl:.2f}")

    final_pnl, final_trades, final_wr = evaluate(agent, eval_env)
    logger.info(f"\nRL Complete | Best: ${best_eval_pnl:.2f} | Final: pnl=${final_pnl:.2f} trades={final_trades} wr={final_wr:.1%}")


if __name__ == "__main__":
    main()
