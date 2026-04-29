"""Scattering Transform + CausalTCN price branch — Phase 4 Candidate B.

Architecture:
    Input (B, T=240, 10)
        ↓
    FeatureSplit:
        - Long-horizon stream: bar_return_bps, spread_pressure  (dim 0,3 of micro block)
        - Short-horizon stream: wick_asymmetry, vol_zscore       (dim 1,2 of micro block)
        - OHLCV passthrough: dims 0-5
        ↓
    LearnableScatteringBlock (long stream + OHLCV, T=240)
        J=3 scales × Q=4 learnable filters per scale
        → scatter_out (B, C_scatter, T//8)
        Captures: spread_pressure envelope + bar_return_bps trend
        Geometric guarantee: ‖S(x) - S(x∘τ)‖ ≤ C‖∇τ‖ (Mallat stability)
        ↓
    CausalTCN (scatter_out, T//8)
        dilated causal conv, 4 layers
        → tcn_out (B, d_model, T//8)
        Captures: long-range temporal dependencies in scattering coefficients
        ↓
    LocalCausalAttention (short stream, last 20 bars)
        window=20, causal mask
        → attn_out (B, d_model)
        Captures: wick_asymmetry bursts + vol_zscore spikes concentrated in last 20 bars
        ↓
    FusionPool: concat + linear → (B, d_model)
        Returns: pooled (B, d_model), seq_features (B, T//8, d_model)  [fusion.py interface]

Evidence base (sequence profiling 240 bars):
    bar_return_bps:   KS 0.12→0.16 (distributed 0-240)  MI=0.015
    wick_asymmetry:   KS 0.22→0.26 (concentrated last 20) MI=0.037
    vol_zscore:       KS ~0.12 (late-window)             MI=0.024
    spread_pressure:  KS 0.50→0.60 (constant offset)    MI=0.056  Cohen's d=-1.2

Backward compat: PriceBranch alias at bottom so fusion.py import works unchanged.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Learnable filterbank ──────────────────────────────────────────────────────

class LearnableFilter(nn.Module):
    """Single learnable bandpass filter — learned FIR, causal, zero-phase init.

    Replaces fixed Morlet wavelets with learned filters whose frequency response
    adapts to XAUUSD M1 timescales during training. Init: approximate Morlet
    at centre_freq so training starts from a principled state.

    Args:
        filter_len: FIR kernel length (odd preferred for symmetric init)
        centre_freq: normalised init frequency in (0, 0.5)
    """
    def __init__(self, filter_len: int = 31, centre_freq: float = 0.1):
        super().__init__()
        assert filter_len % 2 == 1, "filter_len should be odd"
        half = filter_len // 2
        t = torch.arange(-half, half + 1, dtype=torch.float32)
        # Morlet-like init: Gaussian envelope × complex exponential (real part)
        sigma = filter_len / 6.0
        morlet = torch.exp(-0.5 * (t / sigma) ** 2) * torch.cos(2 * math.pi * centre_freq * t)
        morlet = morlet / morlet.norm()
        self.weight = nn.Parameter(morlet.unsqueeze(0).unsqueeze(0))  # (1, 1, L)
        self.filter_len = filter_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — apply filter per channel, causal (left-pad)
        pad = self.filter_len - 1
        x_pad = F.pad(x, (pad, 0))
        w = self.weight.expand(x.shape[1], 1, -1)
        return F.conv1d(x_pad, w, groups=x.shape[1])


# ── Learnable Scattering Block ────────────────────────────────────────────────

class LearnableScatteringBlock(nn.Module):
    """Learnable scattering transform — batched conv1d (Phase 4 compute fix).

    BEFORE: 12 separate F.conv1d calls in a Python loop = 96 CUDA kernel
    launches per batch, each with high launch overhead → 34s/batch observed.

    AFTER: single batched grouped F.conv1d:
        Weight shape: (C * J*Q, 1, L)  — all filters for all channels stacked
        Input shape:  (B, C * J*Q, T)  — input replicated across filter groups
        groups = C * J*Q               — each filter sees exactly one channel
    Result: 1 CUDA kernel launch, same arithmetic, ~19× speedup.

    Geometric stability preserved: the batched conv is mathematically identical
    to the sequential loop — only the execution order changes.

    Args:
        in_channels:  number of input feature channels (default 8)
        J:            frequency scales  (default 3)
        Q:            filters per scale (default 4) → J*Q=12 total filters
        filter_len:   FIR kernel length — must be odd (default 31)
        pool_size:    downsampling stride after filtering (default 2)
        dropout:      applied to layer-0 output (default 0.1)
    """

    def __init__(
        self,
        in_channels: int = 8,
        J: int = 3,
        Q: int = 4,
        filter_len: int = 31,
        pool_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert filter_len % 2 == 1, "filter_len must be odd"
        self.J           = J
        self.Q           = Q
        self.n_filters   = J * Q          # 12
        self.in_channels = in_channels
        self.filter_len  = filter_len
        self.pool_size   = pool_size
        self.pad         = filter_len - 1  # causal left-pad size

        # ── Batched filter bank: (n_filters, 1, L) ───────────────────────────
        # Init each filter as a Morlet wavelet at its centre frequency,
        # then stack into a single parameter tensor — one conv1d replaces the loop.
        half = filter_len // 2
        t    = torch.arange(-half, half + 1, dtype=torch.float32)
        weights = []
        for j in range(J * Q):
            cf     = 0.5 * (j + 1) / (J * Q)
            sigma  = filter_len / 6.0
            morlet = torch.exp(-0.5 * (t / sigma) ** 2) * torch.cos(
                2 * math.pi * cf * t
            )
            morlet = morlet / morlet.norm()
            weights.append(morlet)
        # Shape: (n_filters, 1, L)
        self.filter_bank = nn.Parameter(torch.stack(weights).unsqueeze(1))

        self.pool = nn.AvgPool1d(pool_size, stride=pool_size)
        self.drop = nn.Dropout(dropout)

        # Low-pass envelope — unchanged
        self.lowpass = nn.Conv1d(
            in_channels, in_channels,
            kernel_size = pool_size * 4 + 1,
            padding     = pool_size * 2,
            groups      = in_channels,
            bias        = False,
        )

        self.out_channels = in_channels + in_channels * J * Q

    def _batched_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all J*Q filters to all C channels in ONE conv1d call.

        Strategy:
            1. Repeat x along channel dim: (B, C, T) → (B, C*n_filters, T)
               by interleaving — channel c gets filters 0..n_filters-1
            2. Build grouped weight: (C*n_filters, 1, L) by repeating filter_bank
               for each input channel
            3. Single F.conv1d with groups=C*n_filters
            4. |·| + pool + reshape to (B, C*n_filters, T//p)

        CUDA kernel launches: 1 (was 12 × C = 96)
        """
        B, C, T = x.shape
        F_n     = self.n_filters
        pad     = self.pad

        # Replicate each channel F_n times: (B, C*F_n, T)
        # x[:, c, :] feeds filters c*F_n .. (c+1)*F_n
        x_rep = x.repeat_interleave(F_n, dim=1)            # (B, C*F_n, T)

        # Build weight: repeat filter_bank for each channel
        # filter_bank: (F_n, 1, L) → (C*F_n, 1, L)
        w = self.filter_bank.repeat(C, 1, 1)               # (C*F_n, 1, L)

        # Causal left-pad + single grouped conv
        x_pad = F.pad(x_rep, (pad, 0))                     # (B, C*F_n, T+pad)
        y     = F.conv1d(x_pad, w, groups=C * F_n)         # (B, C*F_n, T)

        y = torch.abs(y)                                    # modulus
        y = self.pool(y)                                    # (B, C*F_n, T//p)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C_in)

        Returns:
            scatter: (B, T//pool_size, out_channels)
        """
        x = x.transpose(1, 2)                              # (B, C, T)

        # Low-pass envelope
        env      = self.lowpass(x)
        env_pool = self.pool(env)                          # (B, C, T//p)

        # All filters in one call — was 12-iteration Python loop
        l0 = self._batched_scatter(x)                      # (B, C*J*Q, T//p)
        l0 = self.drop(l0)

        out = torch.cat([env_pool, l0], dim=1)             # (B, out_channels, T//p)
        return out.transpose(1, 2)                         # (B, T//p, out_channels)


# ── Causal TCN ────────────────────────────────────────────────────────────────

class CausalTCNBlock(nn.Module):
    """Dilated causal TCN block. Operates on scattering coefficients."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop  = nn.Dropout(dropout)
        self._pad  = pad

    def _causal_trim(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :x.shape[2] - self._pad] if self._pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        h = F.gelu(self._causal_trim(self.conv1(x)))
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(self._causal_trim(self.conv2(self.drop(h))))
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        return h + residual


# ── Local Causal Attention ────────────────────────────────────────────────────

class LocalCausalAttention(nn.Module):
    """Memory-efficient causal attention for the short-horizon stream.

    OOM fix: the full (B, H, T, T) score matrix at T=240, B=2048 requires
    ~1.9 GB. Instead:
      1. Only run attention on the last `window` bars (the short stream is
         sliced to T=window before being passed here — see ScatterTCNPriceBranch).
      2. Use F.scaled_dot_product_attention with an is_causal mask, which
         fuses QKV into a single memory-efficient kernel (FlashAttention when
         available, otherwise chunked — both avoid materialising the full (T,T)
         matrix).

    At T=window=20, B=2048, H=4: score matrix = 2048*4*20*20*4B = 13 MB — safe.
    """

    def __init__(self, d_model: int, n_heads: int = 4, window: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.window  = window
        self.d_k     = d_model // n_heads
        self.q   = nn.Linear(d_model, d_model, bias=False)
        self.k   = nn.Linear(d_model, d_model, bias=False)
        self.v   = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop_p  = dropout
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should already be (B, window, D) — sliced by caller
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        Q = self.q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B,H,T,dk)
        K = self.k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Build explicit causal mask — avoids the is_causal=True path that
        # crashes in compiled+cudagraph context on short sequences (T=20).
        # At T=20 the mask is tiny (20×20 = 400 elements) — no memory concern.
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=Q.device, dtype=Q.dtype), diagonal=1
        )
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask = causal_mask,
            dropout_p = self.drop_p if self.training else 0.0,
            is_causal  = False,   # mask supplied explicitly
        )                                                      # (B,H,T,dk)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        return residual + self.out(out)


# ── Main PriceBranch (Scattering + TCN + CausalAttn) ─────────────────────────

class ScatterTCNPriceBranch(nn.Module):
    """Candidate B price encoder: Scattering + CausalTCN + LocalCausalAttention.

    Feature routing (evidence-based from 240-bar sequence profiling):
        Long-horizon stream  → LearnableScattering → CausalTCN
            Features: OHLCV(6) + bar_return_bps(idx6) + spread_pressure(idx9)
            Rationale: KS rises 0.50→0.60 over 240 bars (spread_pressure);
                       bar_return_bps distributed over full sequence.

        Short-horizon stream → LocalCausalAttention(w=20)
            Features: wick_asymmetry(idx7) + vol_zscore(idx8) + bar_return_bps(idx6)
            Rationale: wick_asymmetry/vol_zscore divergence concentrated in last 20 bars.

    Both streams are projected to d_model before fusion.
    Returns (pooled, seq_features) matching original PriceBranch interface.

    Args:
        input_dim:    total input features (default 10 for 6-OHLCV + 4-micro)
        d_model:      shared representation dim (default 192, matches fusion.py)
        scatter_J:    scattering scales (default 3)
        scatter_Q:    filters per scale (default 4)
        filter_len:   FIR kernel length (default 31)
        tcn_layers:   causal TCN depth after scattering (default 4)
        attn_window:  local attention window in bars (default 20)
        dropout:      global dropout rate (default 0.1)

        Legacy fusion.py kwargs accepted and mapped/ignored:
        inception_channels, n_inception_blocks, kernel_sizes,
        tcn_kernel_size, price_dropout, **_kwargs
    """

    # Feature index constants (10-dim input)
    _LONG_IDX  = [0, 1, 2, 3, 4, 5, 6, 9]   # OHLCV(0-5) + bar_return(6) + spread_pressure(9)
    _SHORT_IDX = [6, 7, 8]                    # bar_return(6) + wick_asymmetry(7) + vol_zscore(8)

    def __init__(
        self,
        input_dim:  int   = 10,
        d_model:    int   = 192,
        scatter_J:  int   = 3,
        scatter_Q:  int   = 4,
        filter_len: int   = 31,
        tcn_layers: int   = 4,
        attn_window: int  = 20,
        dropout:    float = 0.1,
        # Legacy kwargs — mapped or ignored
        inception_channels:  int   = None,
        n_inception_blocks:  int   = None,
        kernel_sizes:        list  = None,
        tcn_kernel_size:     int   = None,
        price_dropout:       float = None,
        **_kwargs,
    ):
        super().__init__()
        if price_dropout is not None and dropout == 0.1:
            dropout = price_dropout

        n_long  = len(self._LONG_IDX)   # 8 channels for long stream
        n_short = len(self._SHORT_IDX)  # 3 channels for short stream

        # ── Long-horizon: Scattering + TCN ───────────────────────────────────
        self.scatter = LearnableScatteringBlock(
            in_channels=n_long,
            J=scatter_J,
            Q=scatter_Q,
            filter_len=filter_len,
            pool_size=2,
            dropout=dropout,
        )
        scatter_out_ch = self.scatter.out_channels  # n_long + n_long*J*Q

        # Project scatter output to d_model for TCN
        self.scatter_proj = nn.Linear(scatter_out_ch, d_model)

        self.tcn = nn.Sequential(*[
            CausalTCNBlock(
                channels=d_model,
                kernel_size=3,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(tcn_layers)
        ])

        # Attention pool over TCN output → (B, d_model)
        self.long_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

        # ── Short-horizon: LocalCausalAttention ──────────────────────────────
        self.short_proj = nn.Linear(n_short, d_model)
        self.attn_layers = nn.ModuleList([
            LocalCausalAttention(d_model, n_heads=4, window=attn_window, dropout=dropout)
            for _ in range(2)
        ])
        self.short_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        # Concat long + short pooled → project to d_model
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.d_model     = d_model
        self.out_dim     = d_model
        self.attn_window = attn_window  # stored for forward slice

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim=10)

        Returns:
            pooled:       (B, d_model)     — for fusion.py early-fusion
            seq_features: (B, T_s, d_model) — for cross-attention in fusion.py
                          T_s = T // scatter.pool_size (downsampled by scattering)
        """
        # ── Long-horizon branch ───────────────────────────────────────────────
        x_long  = x[:, :, self._LONG_IDX]           # (B, T, 8)
        scatter = self.scatter(x_long)               # (B, T//2, scatter_out_ch)
        scatter = self.scatter_proj(scatter)         # (B, T//2, d_model)

        # TCN expects (B, C, T)
        tcn_in  = scatter.transpose(1, 2)            # (B, d_model, T//2)
        tcn_out = self.tcn(tcn_in).transpose(1, 2)  # (B, T//2, d_model)

        # Attention-weighted pool → (B, d_model)
        long_scores  = self.long_pool(tcn_out).squeeze(-1)
        long_weights = torch.softmax(long_scores, dim=-1)
        long_pooled  = (tcn_out * long_weights.unsqueeze(-1)).sum(1)

        # ── Short-horizon branch (last `attn_window` bars only) ─────────────
        # Slice BEFORE attention: avoids building (B,H,T=240,T=240) score matrix.
        # Signal is concentrated in last 20 bars (wick_asymmetry, vol_zscore).
        x_short  = x[:, -self.attn_window:, :][:, :, self._SHORT_IDX]  # (B,w,3)
        h_short  = self.short_proj(x_short)           # (B, w, d_model)
        for layer in self.attn_layers:
            h_short = layer(h_short)                  # (B, w, d_model)  T=w=20

        # Pool over the window
        short_scores  = self.short_pool(h_short).squeeze(-1)
        short_weights = torch.softmax(short_scores, dim=-1)
        short_pooled  = (h_short * short_weights.unsqueeze(-1)).sum(1)

        # ── Fusion ────────────────────────────────────────────────────────────
        combined = torch.cat([long_pooled, short_pooled], dim=-1)  # (B, d_model*2)
        pooled   = self.fusion(combined)                           # (B, d_model)

        # seq_features for cross-attention in fusion.py (use TCN output)
        seq_features = tcn_out                                     # (B, T//2, d_model)

        return pooled, seq_features


# ── Backward-compatible alias ─────────────────────────────────────────────────

class PriceBranch(ScatterTCNPriceBranch):
    """Alias so fusion.py `from src.encoder.price_branch_scatter_tcn import PriceBranch` works."""
    pass