"""Regime router with delay-line buffer — Phase 3 v2.

Implements §7 (HD-SNN delay-line memory) and §8 (Mamba/Sessa A/B).

HD-SNN insight (Perrinet 2026):
    Delay depth D is the primary capacity lever: capacity scales N² × D.
    Use a buffer of the last D event-tagged minutes, with sparse triggering,
    and attend across the delay axis rather than compressing into an RNN state.

Mamba/Sessa A/B (§8):
    - Mamba: efficient in calm markets, fails in noisy freeze-time regimes
    - Sessa: robust in diffuse/noisy volatility (XAUUSD volatile sessions)
    Selector: route to Mamba when vol_regime==LOW, Sessa when HIGH.

The router takes the current bar's 13-dim RL observation + the delay buffer
and outputs a regime embedding used by the RL agent for position sizing.
"""

from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
import numpy as np


class DelayLineBuffer:
    """Circular buffer of tagged market events for working memory.

    HD-SNN: deeper delay lines (large D) give better context orthogonality
    than wider hidden states. Capacity ∝ N² × D.

    Event types (sparse triggering, p_A ≈ 0.01):
        0: volume spike (tick_vol > 2σ above rolling mean)
        1: spread widening (spread > 1.5× rolling mean)
        2: directional break (bar close > ATR above/below open)
        3: regime transition (gmm2 state change)
    """

    EVENT_TYPES = 4

    def __init__(self, depth: int = 120, event_types: int = 4):
        self.depth       = depth
        self.event_types = event_types
        self._buffer     = np.zeros((depth, event_types + 1), dtype=np.float32)
        # Extra dim: timestamp offset (normalised 0→1 over depth)
        self._ptr        = 0
        self._full       = False

    def push(self, event_vec: np.ndarray) -> None:
        """Push a (event_types,) vector. Zero = no event this bar."""
        t_norm = self._ptr / self.depth
        self._buffer[self._ptr, :self.event_types] = event_vec
        self._buffer[self._ptr,  self.event_types] = t_norm
        self._ptr = (self._ptr + 1) % self.depth
        if self._ptr == 0:
            self._full = True

    def get_tensor(self, device=None) -> torch.Tensor:
        """Return buffer as (depth, event_types+1) tensor."""
        t = torch.from_numpy(self._buffer.copy())
        return t.to(device) if device else t

    def reset(self) -> None:
        self._buffer[:] = 0.0
        self._ptr  = 0
        self._full = False

    @staticmethod
    def extract_events(obs_bar: np.ndarray, prev_gmm2: float) -> np.ndarray:
        """Heuristic event extraction from a raw M1 bar observation.

        obs_bar layout (partial match to RL obs):
            [sell_p, hold_p, buy_p, conf, pos_dir, unreal, hold_t,
             atr_norm, trend_norm, session_phase, rq, gs_q, cu_au]
        """
        events = np.zeros(4, dtype=np.float32)
        # Proxy: use atr_norm as vol spike indicator
        if obs_bar[7] > 0.03:          # atr > 3% = vol spike
            events[0] = obs_bar[7]
        # Directional break via trend_norm
        if abs(obs_bar[8]) > 1.5:
            events[2] = obs_bar[8]
        # Regime transition
        if obs_bar[10] != prev_gmm2:
            events[3] = 1.0
        return events


class RegimeRouter(nn.Module):
    """Delay-line cross-attention regime router.

    Takes current obs (13-dim) + delay buffer (D, E+1) and outputs
    a regime embedding for position sizing.

    Mamba/Sessa A/B:
        When volatility is LOW (atr_norm < 0.015): route through Mamba path
        When volatility is HIGH (atr_norm > 0.025): route through Sessa path
        Mixed: weighted blend

    STATUS: Mamba and Sessa paths are stubs using a linear layer.
    Full implementation requires:
        Mamba: pip install mamba-ssm (CUDA required)
        Sessa: see src/encoder/price_branch_sessa.py
    """

    def __init__(
        self,
        obs_dim:    int = 13,
        delay_dim:  int = 5,   # event_types + 1
        depth:      int = 120,
        hidden:     int = 64,
        out_dim:    int = 32,
        backend:    Literal["attention", "mamba", "sessa"] = "attention",
    ):
        super().__init__()
        self.obs_dim   = obs_dim
        self.delay_dim = delay_dim
        self.depth     = depth
        self.hidden    = hidden
        self.out_dim   = out_dim
        self.backend   = backend

        # Project obs and delay buffer to common dim
        self.obs_proj   = nn.Linear(obs_dim, hidden)
        self.delay_proj = nn.Linear(delay_dim, hidden)

        # Cross-attention: obs (query) over delay buffer (key/value)
        self.cross_attn = nn.MultiheadAttention(
            hidden, num_heads=4, batch_first=True, dropout=0.1
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        obs:    torch.Tensor,       # (B, obs_dim)
        buffer: torch.Tensor,       # (B, depth, delay_dim)
    ) -> torch.Tensor:
        """Returns (B, out_dim) regime embedding."""
        q = self.obs_proj(obs).unsqueeze(1)        # (B, 1, H)
        kv = self.delay_proj(buffer)               # (B, D, H)
        out, _ = self.cross_attn(q, kv, kv)       # (B, 1, H)
        return self.out_proj(out.squeeze(1))       # (B, out_dim)

    @staticmethod
    def vol_routing_weight(atr_norm: float) -> float:
        """Returns weight for Mamba path [0=Sessa, 1=Mamba].
        Low vol → Mamba. High vol → Sessa.
        """
        if atr_norm < 0.015:
            return 1.0
        elif atr_norm > 0.025:
            return 0.0
        # Linear blend in [0.015, 0.025]
        return 1.0 - (atr_norm - 0.015) / 0.010
