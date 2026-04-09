#!/usr/bin/env python3
"""Train RL agent with confidence-based position sizing.

Optionally uses GAN-simulated markets for training augmentation.

Usage:
    python scripts/train_rl.py --data-dir data --steps 500000
    python scripts/train_rl.py --data-dir data --steps 500000 --use-gan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from loguru import logger

from src.data.preprocessing import WindowMinMaxScaler, prepare_features
from src.data.tick_store import TickStore
from src.meta_policy.rl_agent import ConfidenceSACAgent
from src.meta_policy.gan_market import GANMarketSimulator
from src.utils.config import set_seed, setup_logger

import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Trading environment for RL with confidence input."""

    def __init__(self, features, prices, max_steps=None):
        super().__init__()
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.max_steps = max_steps or len(features) - 1

        feature_dim = features.shape[1]
        self.observation_space = spaces.Box(-np.inf, np.inf, (feature_dim + 3,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        self.step_idx = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.balance = 10_000.0

    def _obs(self):
        feat = self.features[self.step_idx]
        pos_info = np.array([self.position, (self.prices[self.step_idx] - self.entry_price) * self.position / 100 if self.position else 0, self.step_idx / self.max_steps], dtype=np.float32)
        return np.concatenate([feat, pos_info])

    def step(self, action):
        action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        price = self.prices[self.step_idx]
        reward = 0.0

        # Close existing position if direction changes
        if self.position != 0 and np.sign(action_val) != np.sign(self.position):
            pnl = (price - self.entry_price) * np.sign(self.position)
            reward += pnl - 0.02  # spread cost
            self.balance += pnl
            self.position = 0.0

        # Open new position
        if self.position == 0 and abs(action_val) > 0.1:
            self.position = action_val
            self.entry_price = price

        self.step_idx += 1
        done = self.step_idx >= self.max_steps or self.balance <= 0

        if done and self.position != 0:
            pnl = (self.prices[min(self.step_idx, len(self.prices) - 1)] - self.entry_price) * np.sign(self.position)
            reward += pnl
            self.balance += pnl

        return self._obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32), reward, done, False, {"balance": self.balance}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.balance = 10_000.0
        return self._obs(), {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--use-gan", action="store_true", help="Augment with GAN-generated data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="models")
    args = parser.parse_args()

    setup_logger()
    set_seed(args.seed)

    store = TickStore(f"{args.data_dir}/ticks.duckdb")
    df = store.query_ohlcv(args.symbol, "M1")
    store.close()

    if df.is_empty():
        logger.error("No data.")
        return

    features, close = prepare_features(df)
    scaler = WindowMinMaxScaler(120)
    features = scaler.transform(features)[120:]
    prices = close[120:]

    env = TradingEnv(features, prices)
    obs_dim = env.observation_space.shape[0]

    agent = ConfidenceSACAgent(obs_dim=obs_dim)

    # Optional GAN training
    if args.use_gan:
        logger.info("Training GAN market simulator...")
        gan = GANMarketSimulator(seq_len=120, device="cuda" if torch.cuda.is_available() else "cpu")
        # Train GAN on real data
        from torch.utils.data import DataLoader, TensorDataset
        real_seqs = np.array([features[i:i+120] for i in range(0, len(features)-120, 120)])
        ds = TensorDataset(torch.FloatTensor(real_seqs))
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        for epoch in range(50):
            for (batch,) in loader:
                gan.train_step(batch)
            if (epoch + 1) % 10 == 0:
                logger.info(f"GAN epoch {epoch+1}/50")
        gan.save(f"{args.save_dir}/gan_market.pt")
        logger.info("GAN training complete")

    # Train RL agent
    obs, _ = env.reset()
    best_reward = -float("inf")
    episode_rewards = []
    ep_reward = 0

    for step in range(1, args.steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, done, _, info = env.step(np.array([action]))

        agent.store(obs, action, reward, next_obs, done)
        ep_reward += reward
        obs = next_obs

        if len(agent.obs_buf[:agent.buffer_size]) >= agent.batch_size:
            agent.update()

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0
            obs, _ = env.reset()

        if step % 10_000 == 0:
            avg = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            logger.info(f"Step {step}/{args.steps} | Avg reward: {avg:.2f} | Balance: {info['balance']:.2f}")
            if avg > best_reward and episode_rewards:
                best_reward = avg
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                agent.save(f"{args.save_dir}/rl_agent_best.pt")

    logger.info(f"RL training complete. Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
