"""Hierarchical RL agent with confidence-based position sizing.

The supervised dual-branch model produces signal + confidence.
The RL agent uses confidence to modulate position size:
    - High confidence → larger position
    - Low confidence → smaller position or skip

Architecture:
    Observation: [model_features, confidence, position_state]
    Action: continuous position size [-1, +1] (scaled by confidence)
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class ConfidenceSACAgent:
    """SAC agent that uses model confidence for position sizing.

    The agent receives:
        - Feature vector from the supervised model
        - Confidence score from the confidence head
        - Current position state (direction, PnL, hold time)

    And outputs a continuous position size: [-1, +1] where:
        - -1 = max short
        - 0 = flat
        - +1 = max long

    The actual lot size = |action| × confidence × max_lots.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list[int] = [256, 256],
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        buffer_capacity: int = 500_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor: obs → continuous action [-1, 1]
        self.actor = self._build_net(obs_dim, 1, hidden_dims, output_act="tanh").to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Twin critics
        self.q1 = self._build_net(obs_dim + 1, 1, hidden_dims).to(self.device)
        self.q2 = self._build_net(obs_dim + 1, 1, hidden_dims).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.q_optim = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        # Replay buffer (pre-allocated numpy)
        self.buffer_pos = 0
        self.buffer_size = 0
        self.buffer_cap = buffer_capacity
        self.obs_buf = np.zeros((buffer_capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_capacity, 1), dtype=np.float32)
        self.rew_buf = np.zeros(buffer_capacity, dtype=np.float32)
        self.next_obs_buf = np.zeros((buffer_capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(buffer_capacity, dtype=np.float32)

        self.train_step = 0

    def _build_net(self, in_dim, out_dim, hidden, output_act=None):
        layers = []
        d = in_dim
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU()])
            d = h
        layers.append(nn.Linear(d, out_dim))
        if output_act == "tanh":
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def select_action(self, obs: np.ndarray, confidence: float = 1.0, eval_mode: bool = False) -> float:
        """Select continuous position size.

        Returns:
            Position size in [-1, 1], modulated by confidence.
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_action = self.actor(obs_t).item()

        if not eval_mode:
            raw_action += np.random.normal(0, 0.2)  # exploration noise
            raw_action = np.clip(raw_action, -1, 1)

        return raw_action * confidence  # modulate by model confidence

    def store(self, obs, action, reward, next_obs, done):
        i = self.buffer_pos
        self.obs_buf[i] = obs
        self.act_buf[i] = action
        self.rew_buf[i] = reward
        self.next_obs_buf[i] = next_obs
        self.done_buf[i] = float(done)
        self.buffer_pos = (self.buffer_pos + 1) % self.buffer_cap
        self.buffer_size = min(self.buffer_size + 1, self.buffer_cap)

    def update(self) -> dict:
        if self.buffer_size < self.batch_size:
            return {}

        idx = np.random.randint(0, self.buffer_size, self.batch_size)
        obs = torch.FloatTensor(self.obs_buf[idx]).to(self.device)
        act = torch.FloatTensor(self.act_buf[idx]).to(self.device)
        rew = torch.FloatTensor(self.rew_buf[idx]).unsqueeze(1).to(self.device)
        nobs = torch.FloatTensor(self.next_obs_buf[idx]).to(self.device)
        done = torch.FloatTensor(self.done_buf[idx]).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_act = self.actor(nobs)
            nq1 = self.q1_target(torch.cat([nobs, next_act], -1))
            nq2 = self.q2_target(torch.cat([nobs, next_act], -1))
            target = rew + (1 - done) * self.gamma * torch.min(nq1, nq2)

        sa = torch.cat([obs, act], -1)
        q1_loss = F.mse_loss(self.q1(sa), target)
        q2_loss = F.mse_loss(self.q2(sa), target)

        self.q_optim.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optim.step()

        # Actor update
        a = self.actor(obs)
        actor_loss = -self.q1(torch.cat([obs, a], -1)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update targets
        for sp, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        for sp, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        self.train_step += 1
        return {"q_loss": (q1_loss + q2_loss).item(), "actor_loss": actor_loss.item()}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.q1.load_state_dict(d["q1"])
        self.q2.load_state_dict(d["q2"])
