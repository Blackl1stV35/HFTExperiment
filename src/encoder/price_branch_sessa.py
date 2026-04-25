"""Sessa price branch stub — Phase 3 v2.

Sessa (Horbatko 2026): Selective State Space Attention.
Sits between standard attention (diffuse → O(1/S_eff) memory) and Mamba
(exponential decay in noisy freeze-time-failed regimes).
Sessa achieves power-law O(ℓ^-β) memory tail via causal triangular solve
(I - B_fb)s = f, making it robust in volatile XAUUSD sessions.

Reference: https://github.com/LibratioAI/sessa

STATUS: Stub — wired to fall back to standard PriceBranch.
To implement: replace SessaLayer.forward() with the triangular solve.
Key operation: torch.linalg.solve_triangular(L, f, upper=False)
where L is the strictly-lower-triangular feedback matrix B_fb.

A/B test against PriceBranch by setting encoder.use_sessa=true in config.
"""

from __future__ import annotations
import warnings
import torch
import torch.nn as nn
from src.encoder.price_branch import PriceBranch


class SessaLayer(nn.Module):
    """Placeholder for the Sessa causal triangular-solve mixer.

    Implements the API of LocalCausalAttention but falls back to standard
    attention until the triangular-solve path is implemented.

    TODO: implement forward path:
        1. Compute forward attention features f = QK^T V (standard)
        2. Build strictly-lower-triangular B_fb from learned parameters
        3. Solve (I - B_fb) s = f via torch.linalg.solve_triangular
        4. Apply scalar gain γ_t = tanh(⟨a_t, w_γ⟩ + b_γ) ∈ (-1, 1)
        5. Return s + residual
    """
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        warnings.warn("SessaLayer is a stub — using standard attention", stacklevel=2)
        from src.encoder.price_branch import LocalCausalAttention
        self._attn = LocalCausalAttention(d_model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._attn(x)


class PriceBranchSessa(PriceBranch):
    """Sessa-variant price branch — A/B against PriceBranch."""

    def __init__(self, in_features=6, hidden_dim=128, n_layers=2,
                 n_heads=4, attn_window=20, dropout=0.1):
        super().__init__(in_features, hidden_dim, n_layers, n_heads,
                         attn_window, dropout)
        # Replace attention layers with Sessa layers
        from torch.nn import ModuleList
        self.stage2 = ModuleList([
            SessaLayer(hidden_dim, n_heads=n_heads,
                       window=attn_window, dropout=dropout)
            for _ in range(n_layers)
        ])
