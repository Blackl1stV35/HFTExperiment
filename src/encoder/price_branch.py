"""Price branch encoder — Phase 3 v2 (CCSO two-stage architecture).

Architecture per CCSO §1.2–1.3:
    Stage 1 — Cross-feature Conv1d  (inter-sequence, feature-axis)
    Stage 2 — Local Attention       (temporal, window w≈20 M1 bars)

CCSO ablation (their Figure 3) shows performance peaks then degrades
as attention window grows beyond the regime-typical pattern length.
Default w=20; sweep as hyperparameter.

Sessa alternative: see price_branch_sessa.py for the power-law memory
tail mixer that outperforms standard attention in diffuse/noisy regimes.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFeatureConv(nn.Module):
    """Stage 1: conv along the feature axis.

    Learns which OHLCV combinations matter (bid-ask × volume relationships
    are roughly stable intra-day) before attending over time.
    """
    def __init__(self, in_features: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        # Conv1d with kernel=1 mixes features at each time step
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


class LocalCausalAttention(nn.Module):
    """Stage 2: causal local attention with window w.

    CCSO: use sliding-window mask with w≈20 for 240-step time-series.
    Complexity drops O(n²) → O(n·w²). Causal (no future leak).
    """
    def __init__(self, d_model: int, n_heads: int = 4,
                 window: int = 20, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.window  = window
        self.d_k     = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        Q = self.q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B,H,T,dk)
        K = self.k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,H,T,T)

        # Causal local mask: allow only positions i-window <= j <= i
        mask = torch.full((T, T), float('-inf'), device=x.device)
        for i in range(T):
            lo = max(0, i - self.window + 1)
            mask[i, lo : i + 1] = 0.0
        scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn = self.drop(torch.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V)                     # (B,H,T,dk)
        out  = out.transpose(1, 2).contiguous().view(B, T, D)
        return residual + self.out(out)


class PriceBranch(nn.Module):
    """Two-stage price encoder: CrossFeatureConv → LocalCausalAttention.

    CCSO §1.2–1.3: Stage 1 mixes features (cross-feature Conv), Stage 2 mixes
    time with a causal local-attention window.

    Accepts both CCSO-style kwargs and the legacy fusion.py kwargs
    (input_dim, inception_channels, tcn_layers, d_model, etc.) so that
    DualBranchModel.from_config() works without modification.

    Legacy → CCSO mapping:
        input_dim          → in_features
        d_model            → hidden_dim  (output dim passed to fusion)
        n_inception_blocks → n_layers
        tcn_layers         → n_layers  (max of the two)
        inception_channels, tcn_kernel_size, kernel_sizes → ignored
    """
    def __init__(
        self,
        # CCSO-style args (preferred)
        in_features: int   = 6,
        hidden_dim:  int   = 128,
        n_layers:    int   = 2,
        n_heads:     int   = 4,
        attn_window: int   = 20,
        dropout:     float = 0.1,
        # Legacy fusion.py kwargs — mapped to CCSO equivalents
        input_dim:          int   = None,
        d_model:            int   = None,
        inception_channels: int   = None,   # absorbed into hidden_dim
        n_inception_blocks: int   = None,   # alias for n_layers
        tcn_layers:         int   = None,   # alias for n_layers
        tcn_kernel_size:    int   = None,   # ignored
        kernel_sizes:       list  = None,   # ignored
        price_dropout:      float = None,   # alias for dropout
        **_kwargs,                          # absorb any remaining unknown args
    ):
        super().__init__()
        # Apply legacy aliases — explicit CCSO args take precedence
        if input_dim is not None and in_features == 6:
            in_features = input_dim
        if d_model is not None and hidden_dim == 128:
            hidden_dim = d_model
        if price_dropout is not None and dropout == 0.1:
            dropout = price_dropout
        if n_inception_blocks is not None:
            n_layers = max(n_layers, n_inception_blocks)
        if tcn_layers is not None:
            n_layers = max(n_layers, tcn_layers)

        self.stage1 = CrossFeatureConv(in_features, hidden_dim, dropout)
        self.stage2 = nn.ModuleList([
            LocalCausalAttention(hidden_dim, n_heads, attn_window, dropout)
            for _ in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.stage1(x)           # (B, T, hidden)
        for layer in self.stage2:
            x = layer(x)             # (B, T, hidden)
        # Pool over time → (B, hidden)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return x