"""LeWM (LeWorldModel) latent world model stub — Phase 3 v2.

Implements §4.2 of IMPROVEMENTS.md (LeWM, Maes et al. 2026):
    Replaces GAN market simulation with a JEPA-style latent world model
    that has provable anti-collapse guarantees (GAN does not).

Architecture:
    Encoder:   dual-branch encoder → latent z_t   (already exists)
    Predictor: small transformer   → ẑ_{t+1} from z_t + action a_t
    Regulariser: SIGReg            → project latents onto M=1024 random
                                     directions, apply Epps-Pulley normality
                                     test on each projection (Cramér-Wold:
                                     if all marginals Gaussian, joint Gaussian)
    Planner:   CEM in latent space → optimise a_{1:H} for max expected return

Losses:
    L_total = L_predict_next + λ * L_sigreg
    L_predict_next = MSE(ẑ_{t+1}, z_{t+1})  (stop-grad on target)
    L_sigreg       = mean KL of projected latents from N(0,1)

Reference: https://github.com/[LeWM repo — see paper for URL]
Paper: Maes et al. 2026, "LeWorldModel: Stable End-to-End JEPA from Pixels"

STATUS: Stub with SIGReg implemented. Predictor is a 2-layer MLP.
TODO: replace predictor with a small causal transformer for multi-step rollout.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictor(nn.Module):
    """Predicts next latent z_{t+1} from current z_t and action a_t.

    TODO: replace with causal transformer for multi-step horizon.
    """
    def __init__(self, latent_dim: int, action_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


class SIGReg(nn.Module):
    """SIGReg: Gaussian regulariser via random projections.

    Projects latents onto M random directions, applies Epps-Pulley test.
    By Cramér-Wold: if all 1-D marginals are N(0,1), the joint is Gaussian.
    This is the anti-collapse mechanism LeWM uses instead of GAN discriminator.

    Implementation: approximate by penalising skewness + excess kurtosis
    on each projection (sufficient statistics for the Epps-Pulley statistic).
    """
    def __init__(self, latent_dim: int, n_projections: int = 256):
        super().__init__()
        self.register_buffer(
            "directions",
            F.normalize(torch.randn(n_projections, latent_dim), dim=-1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D)
        # Project: (B, M)
        proj = z @ self.directions.T
        # Normalise each projection
        proj = (proj - proj.mean(0)) / (proj.std(0) + 1e-8)
        # Penalise skewness^2 + excess kurtosis^2 (Epps-Pulley approximation)
        skew = (proj ** 3).mean(0)
        kurt = (proj ** 4).mean(0) - 3.0
        return (skew ** 2 + kurt ** 2).mean()


class LeWMWorldModel(nn.Module):
    """Latent world model for RL data augmentation.

    Training:
        model = LeWMWorldModel(latent_dim=256)
        # Forward pass
        z_pred = model.predict(z_t, action_onehot)
        loss   = model.loss(z_pred, z_t_plus_1)
        loss.backward()

    Planning (CEM in latent space):
        best_actions = model.plan_cem(z_0, horizon=5, n_samples=64)
    """

    def __init__(
        self,
        latent_dim:     int = 256,
        action_dim:     int = 3,    # long / flat / short
        n_projections:  int = 256,
        sigreg_lambda:  float = 0.1,
    ):
        super().__init__()
        self.predictor     = LatentPredictor(latent_dim, action_dim)
        self.sigreg        = SIGReg(latent_dim, n_projections)
        self.sigreg_lambda = sigreg_lambda
        self.latent_dim    = latent_dim
        self.action_dim    = action_dim

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, action)

    def loss(
        self,
        z_pred:    torch.Tensor,
        z_target:  torch.Tensor,   # stop-gradient from encoder
        z_context: torch.Tensor,   # current batch latents for SIGReg
    ) -> torch.Tensor:
        l_pred   = F.mse_loss(z_pred, z_target.detach())
        l_sigreg = self.sigreg(z_context)
        return l_pred + self.sigreg_lambda * l_sigreg

    @torch.no_grad()
    def plan_cem(
        self,
        z0:        torch.Tensor,
        horizon:   int = 5,
        n_samples: int = 64,
        n_elite:   int = 10,
        n_iters:   int = 5,
    ) -> torch.Tensor:
        """Cross-Entropy Method planning in latent space.

        Optimises action sequence a_{0:H} to minimise predicted latent
        distance from a "goal" embedding. For trading, goal = highest
        predicted return direction.

        Returns: (H,) tensor of discrete action indices.
        """
        device = z0.device
        # Initialise with uniform categorical
        probs = torch.ones(horizon, self.action_dim, device=device) / self.action_dim

        for _ in range(n_iters):
            # Sample action sequences
            actions = torch.zeros(n_samples, horizon, dtype=torch.long, device=device)
            for h in range(horizon):
                dist = torch.distributions.Categorical(probs[h])
                actions[:, h] = dist.sample((n_samples,))

            # Roll out in latent space
            z = z0.unsqueeze(0).expand(n_samples, -1)
            rewards = torch.zeros(n_samples, device=device)
            for h in range(horizon):
                a_oh = F.one_hot(actions[:, h], self.action_dim).float()
                z    = self.predictor(z, a_oh)
                # Proxy reward: norm of latent (higher = more signal)
                rewards += z.norm(dim=-1)

            # Elite update
            elite_idx = rewards.topk(n_elite).indices
            elite_actions = actions[elite_idx]   # (n_elite, H)
            for h in range(horizon):
                counts = torch.bincount(elite_actions[:, h], minlength=self.action_dim).float()
                probs[h] = counts / counts.sum()

        return probs.argmax(dim=-1)   # (H,) best action sequence
