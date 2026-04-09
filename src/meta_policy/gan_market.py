"""GAN-simulated market generator for RL training generalization.

Generates synthetic XAUUSD-like price sequences that preserve the statistical
properties of real data (volatility clustering, fat tails, mean reversion)
while introducing controlled variation to prevent RL overfitting.

Architecture:
    Generator: noise → LSTM → synthetic OHLCV sequences
    Discriminator: sequence → real/fake classification
    Training: Wasserstein GAN-GP for stable training
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class MarketGenerator(nn.Module):
    """LSTM-based generator for synthetic OHLCV sequences."""

    def __init__(self, noise_dim: int = 32, hidden_dim: int = 128, output_dim: int = 6, seq_len: int = 120):
        super().__init__()
        self.noise_dim = noise_dim
        self.seq_len = seq_len

        self.fc_in = nn.Linear(noise_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, noise_dim) → (batch, seq_len, output_dim)"""
        batch = z.shape[0]
        h = self.fc_in(z).unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.lstm(h)
        return self.act(self.fc_out(out))


class MarketDiscriminator(nn.Module):
    """1D-CNN discriminator for real vs synthetic sequences."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, features) → (batch, 1)"""
        return self.net(x.permute(0, 2, 1))


class GANMarketSimulator:
    """Wasserstein GAN-GP trainer for synthetic market generation."""

    def __init__(
        self,
        seq_len: int = 120,
        noise_dim: int = 32,
        feature_dim: int = 6,
        lr: float = 1e-4,
        gp_weight: float = 10.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.noise_dim = noise_dim
        self.gp_weight = gp_weight

        self.generator = MarketGenerator(noise_dim, 128, feature_dim, seq_len).to(self.device)
        self.discriminator = MarketDiscriminator(feature_dim).to(self.device)

        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.0, 0.9))
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        batch = real.shape[0]
        alpha = torch.rand(batch, 1, 1, device=self.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = self.discriminator(interp)
        grads = torch.autograd.grad(
            d_interp, interp, grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        gp = ((grads.norm(2, dim=(1, 2)) - 1) ** 2).mean()
        return gp

    def train_step(self, real_data: torch.Tensor, n_critic: int = 5) -> dict:
        """One training step. Returns losses."""
        real = real_data.to(self.device)
        batch = real.shape[0]

        # Train discriminator
        for _ in range(n_critic):
            z = torch.randn(batch, self.noise_dim, device=self.device)
            fake = self.generator(z).detach()
            d_real = self.discriminator(real).mean()
            d_fake = self.discriminator(fake).mean()
            gp = self._gradient_penalty(real, fake)
            d_loss = d_fake - d_real + self.gp_weight * gp

            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

        # Train generator
        z = torch.randn(batch, self.noise_dim, device=self.device)
        fake = self.generator(z)
        g_loss = -self.discriminator(fake).mean()

        self.g_optim.zero_grad()
        g_loss.backward()
        self.g_optim.step()

        return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic market sequences."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.noise_dim, device=self.device)
            synthetic = self.generator(z).cpu().numpy()
        return synthetic

    def save(self, path: str) -> None:
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(data["generator"])
        self.discriminator.load_state_dict(data["discriminator"])
