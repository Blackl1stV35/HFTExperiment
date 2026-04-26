"""CCSO two-stage price branch — Phase 3 v2.

Architecture per CCSO §1.2–1.3:
    Stage 1 — CrossFeatureConv  (inter-feature mixing, feature axis)
    Stage 2 — LocalCausalAttention  (temporal, causal window w=20)

CCSO ablation (Fig 3): performance peaks then degrades as attention window
grows beyond the regime-typical pattern length. Default w=20 M1 bars.
O(n·w²) complexity vs O(n²) for full attention.

This file is the CCSO variant. The original InceptionBlock+TCN architecture
(which matches the trained checkpoint) is in price_branch.py.

To activate: change fusion.py import:
    from src.encoder.price_branch_ccso import PriceBranch
Then retrain from scratch — weight keys differ from the InceptionBlock checkpoint.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Stage 1: CrossFeatureConv ─────────────────────────────────────────────────

class CrossFeatureConv(nn.Module):
    """Mix the feature axis (OHLCV combinations) at each time step.

    CCSO §1.2: inter-sequence (cross-feature) convolution first.
    Bid-ask × volume relationships are roughly stable intra-day, so learning
    feature interactions before attending over time is more sample-efficient.

    Args:
        in_features: input feature dim (default 6 for OHLCV+spread)
        hidden:      output hidden dim
        dropout:     applied after second linear
    """

    def __init__(self, in_features: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1  = nn.Linear(in_features, hidden * 2)
        self.fc2  = nn.Linear(hidden * 2, hidden)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, hidden) if in_features != hidden else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        residual = self.proj(x)
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x + residual


# ── Stage 2: LocalCausalAttention ─────────────────────────────────────────────

class LocalCausalAttention(nn.Module):
    """Causal local attention with sliding window w.

    CCSO §1.3: use window w≈20 for M1 bars. Complexity O(n·w²) not O(n²).
    Causal mask: position i can only attend to [i-w+1, i].

    Args:
        d_model:  model dimension
        n_heads:  attention heads (d_model must be divisible)
        window:   causal window size in bars (default 20)
        dropout:  applied to attention weights
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int   = 4,
        window:   int   = 20,
        dropout:  float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.window  = window
        self.d_k     = d_model // n_heads

        self.q   = nn.Linear(d_model, d_model, bias=False)
        self.k   = nn.Linear(d_model, d_model, bias=False)
        self.v   = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D  = x.shape
        residual = x
        x = self.norm(x)

        Q = self.q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, dk)
        K = self.k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, T)

        # Causal local mask: i can attend to [max(0, i-window+1) .. i]
        mask = torch.full((T, T), float('-inf'), device=x.device, dtype=x.dtype)
        for i in range(T):
            lo = max(0, i - self.window + 1)
            mask[i, lo : i + 1] = 0.0
        scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn = self.drop(torch.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V)                          # (B, H, T, dk)
        out  = out.transpose(1, 2).contiguous().view(B, T, D)
        return residual + self.out(out)


# ── PriceBranch ───────────────────────────────────────────────────────────────

class PriceBranch(nn.Module):
    """CCSO two-stage price encoder.

    Stage 1: CrossFeatureConv  — learns OHLCV feature interactions
    Stage 2: LocalCausalAttention × n_layers  — learns temporal patterns
    Output: (B, d_model) pooled sequence representation

    Also returns seq_features (B, T, d_model) for cross-attention in fusion.py,
    matching the original PriceBranch interface.

    Args:
        input_dim:          OHLCV feature dim (default 6)
        d_model:            output dim (default 192, matches fusion.py default)
        n_inception_blocks: mapped to n_layers for CCSO (legacy compat)
        tcn_layers:         mapped to n_layers for CCSO (legacy compat)
        n_heads:            attention heads in stage 2 (default 4)
        attn_window:        local attention window in bars (default 20)
        inception_channels: ignored (CCSO uses d_model throughout)
        tcn_kernel_size:    ignored
        kernel_sizes:       ignored
        dropout:            applied in both stages
    """

    def __init__(
        self,
        # Primary args
        input_dim:          int   = 6,
        d_model:            int   = 192,
        n_heads:            int   = 4,
        attn_window:        int   = 20,
        dropout:            float = 0.1,
        # n_layers — or legacy aliases
        n_layers:           int   = 2,
        n_inception_blocks: int   = None,   # alias → n_layers
        tcn_layers:         int   = None,   # alias → n_layers
        # Ignored legacy kwargs — accepted to avoid TypeError from fusion.py
        inception_channels: int   = None,
        tcn_kernel_size:    int   = None,
        kernel_sizes:       list  = None,
        **_kwargs,
    ):
        super().__init__()

        # Resolve n_layers from any alias
        if n_inception_blocks is not None:
            n_layers = max(n_layers, n_inception_blocks)
        if tcn_layers is not None:
            n_layers = max(n_layers, tcn_layers)

        self.d_model = d_model

        self.stage1 = CrossFeatureConv(input_dim, d_model, dropout)
        self.stage2 = nn.ModuleList([
            LocalCausalAttention(d_model, n_heads, attn_window, dropout)
            for _ in range(n_layers)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim)

        Returns:
            pooled:       (B, d_model)  — global sequence representation
            seq_features: (B, T, d_model) — per-timestep features for cross-attn
        """
        # Stage 1: cross-feature mixing
        x = self.stage1(x)              # (B, T, d_model)

        # Stage 2: local causal temporal attention
        for layer in self.stage2:
            x = layer(x)               # (B, T, d_model)

        seq_features = self.norm(x)    # (B, T, d_model)

        # Attention-weighted global pooling
        attn_scores  = self.attn_pool(seq_features).squeeze(-1)   # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled       = (seq_features * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        return pooled, seq_features
