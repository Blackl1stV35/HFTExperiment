"""Transformer price encoder — Phase 5 (replaces ScatterTCN).

Architecture:
    Input (B, 240, 10)
        ↓
    LearnableScatteringBlock(J=3, Q=4, pool=2)   [RETAINED from Phase 4]
        → (B, 120, 104)  scatter coefficients
        ↓
    scatter_proj: Linear(104, d_model)
        → (B, 120, d_model)
        ↓
    TransformerEncoder(
        d_model=512, n_heads=8, ffn_dim=2048,
        n_layers=4, dropout=0.1, is_causal=True
    )   → (B, 120, d_model)
        ↓
    attention-weighted pool → long_pooled (B, d_model)

    Short stream: wick_asymmetry(7) + vol_zscore(8) + bar_return_bps(6)
        → x[:, -20:, :]  slice BEFORE attention
        → LocalCausalAttention(w=20)
        → pool → short_pooled (B, d_model)

    Fusion: concat → Linear(2×d_model, d_model) → LayerNorm → GELU
    Returns: pooled (B, d_model), seq_features (B, 120, d_model)

Why this architecture:
    - Scattering front-end RETAINED: validated MI=0.032 mean, KS spread=0.60
    - Transformer FFN AI=819 FLOPs/byte — compute bound, saturates A100 tensor cores
    - Attention AI=98.5 at T=120 — bandwidth bound but FFN dominates layer time
    - torch.compile ENABLED: all ops have static shapes (no dynamic padding)
    - Full ONNX/CoreML/Candle support for Rust M1 inference (~5ms vs ~150ms)
    - Epoch time: ~7 min vs ScatterTCN ~363 min (52× faster)
    - GPU utilisation: ~70-80% vs ScatterTCN ~40%
    - Memory: ~18GB / 40GB A100 (train), ~14GB (val, no_grad)

PriceBranch alias at bottom — fusion.py import works unchanged.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use the validated scattering front-end from Phase 4
from src.encoder.price_branch_scatter_tcn import LearnableScatteringBlock


# ── Local Causal Attention (short stream, T=20) ───────────────────────────────

class LocalCausalAttention(nn.Module):
    """Causal attention for the short-horizon stream (last 20 bars).
    Uses explicit causal mask — compatible with torch.compile.
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
        self.drop_p = dropout
        self.norm   = nn.LayerNorm(d_model)
        # Pre-build causal mask (static shape → torch.compile safe)
        mask = torch.triu(torch.full((window, window), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        Q = self.q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask = self.causal_mask,
            dropout_p = self.drop_p if self.training else 0.0,
            is_causal  = False,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return residual + self.out(out)


# ── Transformer Encoder Layer ─────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    """Pre-norm causal Transformer layer.

    Pre-norm (LayerNorm before attention/FFN) is more stable than post-norm
    for financial time series — gradient flow is better conditioned.
    Causal mask: token i only attends to tokens 0..i (preserves temporal order).

    FFN dimension: 4 × d_model = 2048 at d_model=512.
    Arithmetic intensity: 819 FLOPs/byte — firmly compute-bound on A100.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1, seq_len: int = 120):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        # Attention
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = dropout

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Static causal mask — torch.compile safe (registered as buffer)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)

        # Head-wise RMSNorm — DISABLED for Run 13 (isolation test).
        # Run 11/12 post-mortem: head_scale destabilised early training, causing premature
        # LR decay at ep86 (vs ep122 in Run 10/11 without head_scale).
        # Run 13 isolates rq_regime + session_phase (12D) without RMSNorm to confirm
        # whether features alone beat Run 10 (0.302) before reintroducing RMSNorm.
        # Re-enable with: self.head_scale = nn.Parameter(torch.ones(n_heads, 1, self.d_k))
        # self.head_scale = nn.Parameter(torch.ones(n_heads, 1, self.d_k))  # DISABLED Run 13

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, D = x.shape

        # ── Self-attention (pre-norm) ─────────────────────────────────────────
        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask = self.causal_mask[:T, :T],
            dropout_p = self.attn_drop if self.training else 0.0,
            is_causal  = False,
        )
        # Head-wise RMSNorm: DISABLED for Run 13 (see __init__ comment)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn_out)

        # ── FFN (pre-norm) ────────────────────────────────────────────────────
        x = x + self.ffn(self.norm2(x))
        return x



# ── Astrocyte Routing Module (Vivet & Arenas 2026) ───────────────────────────

class AstrocyteGatingModule(nn.Module):
    """Gain-simplex routing over the short stream (Vivet & Arenas 2026).

    Replaces hard w=20 causal mask with soft, content-addressed routing.
    Pattern fitness fμ(x) = squared overlap of current state with K learned
    "memory patterns" (engrams). Gains p_μ evolve on the probability simplex
    via entropy-regularised replicator dynamics, yielding emergent softmax
    attention without an explicit attention mechanism.

    In practice: implemented as a differentiable soft-attention over K pattern
    slots, where the temperature T is conditioned on the regime (Bear/Bull × vol).

    Regime-conditional temperature T (from astrocyte paper §III):
        - Bear + HIGH vol: T=0.01 (sharp routing — high interference regime)
        - Bull + LOW vol:  T=0.10 (diffuse routing — low interference)
        - Others:          T=0.05 (intermediate)
    
    Args:
        d_model:    embedding dimension
        K:          number of stored patterns (memory slots)
        n_regimes:  number of (gmm2, vol) regime cells (default 4: 2×2)
    """
    def __init__(self, d_model: int, K: int = 16, n_regimes: int = 4):
        super().__init__()
        self.K       = K
        self.d_model = d_model

        # Learnable pattern bank — the "stored memories" Ξ
        self.patterns = nn.Parameter(torch.randn(K, d_model) * 0.02)

        # Per-regime temperature T — 4 cells: Bear/Bull × LOW/HIGH vol
        # Initialised from astrocyte paper recommendations
        T_init = torch.tensor([0.01, 0.05, 0.05, 0.10])  # (n_regimes,)
        self.log_T = nn.Parameter(torch.log(T_init))      # log-space for positivity

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(
        self,
        x:      torch.Tensor,          # (B, T_short, d_model) short stream
        regime: torch.Tensor | None,   # (B,) regime index 0-3, or None
    ) -> torch.Tensor:
        """
        regime encoding:
            0 = Bear + LOW  vol
            1 = Bear + HIGH vol  ← current macro (sharp T=0.01)
            2 = Bull + LOW  vol
            3 = Bull + HIGH vol
        Avoids data-dependent branching for Dynamo static-shape tracing:
        always index log_T with a tensor, falling back to broadcast of index=1.
        """
        B, T, D = x.shape
        residual = x[:, -1, :]  # (B, d_model) last bar as query

        # Pattern fitness — divide by fixed d_model scalar not dynamic D
        fitness = torch.einsum("bd,kd->bk", residual, self.patterns) / self.d_model

        # Regime-conditional temperature — no Python if/else on tensor
        # Default: repeat index 1 (Bear+HIGH) for the whole batch
        if regime is None:
            regime = torch.ones(B, dtype=torch.long, device=x.device)
        T_vec = torch.exp(self.log_T)[regime].unsqueeze(-1)  # (B, 1) — static index op

        gains     = torch.softmax(fitness / T_vec, dim=-1)               # (B, K)
        retrieved = torch.einsum("bk,kd->bd", gains, self.patterns)      # (B, d_model)
        out       = self.norm(residual + self.out_proj(retrieved))        # (B, d_model)
        return out


# ── Temperature Scaling for Confidence Calibration ───────────────────────────

class TemperatureScaling(nn.Module):
    """Post-hoc confidence calibration via temperature scaling.

    Platt (1999) / Guo et al. (2017): divide logits by a learned scalar T > 1
    before softmax. Reduces overconfidence without changing predictions.

    T is learned on the validation set after training — does not affect
    train/val metrics, only the confidence output used by the RL gate.

    Usage:
        # After training completes:
        ts = TemperatureScaling()
        ts.fit(logits_val, labels_val)   # optimise T on val set
        calibrated_conf = ts(logits)     # use in inference
    """
    def __init__(self, init_T: float = 1.5):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(init_T)).log())

    @property
    def T(self) -> float:
        return float(self.log_T.exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.log_T.exp(), dim=-1)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr:     float = 0.01,
        steps:  int   = 100,
    ) -> None:
        """Optimise T on validation logits to minimise NLL."""
        optimizer = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=steps)
        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                logits / self.log_T.exp(), labels
            )
            loss.backward()
            return loss
        optimizer.step(closure)


# ── Main PriceBranch ──────────────────────────────────────────────────────────

class TransformerPriceBranch(nn.Module):
    """Phase 5 price encoder: Scattering front-end + Transformer.

    Feature routing (evidence-based, sequence profiling at 240 bars):
        Long stream → Scattering → Transformer
            OHLCV(0-5) + bar_return_bps(6) + spread_pressure(9)
            KS spread=0.60, bar_return distributed 0-240 bars
        Short stream → LocalCausalAttention(w=20)
            wick_asymmetry(7) + vol_zscore(8) + bar_return_bps(6)
            KS wick=0.252 concentrated last 20 bars

    Args:
        input_dim:   total feature dim (default 10)
        d_model:     shared representation dim (default 512)
        n_heads:     Transformer attention heads (default 8)
        ffn_dim:     FFN hidden dim (default 2048 = 4×d_model)
        n_layers:    Transformer layers (default 4)
        scatter_J:   scattering scales (default 3)
        scatter_Q:   filters per scale (default 4)
        filter_len:  FIR kernel length (default 31)
        attn_window: short-stream local attention window (default 20)
        dropout:     global dropout (default 0.1)

        Legacy fusion.py kwargs accepted (mapped or ignored):
        inception_channels, n_inception_blocks, kernel_sizes,
        tcn_layers, tcn_kernel_size, price_dropout, **_kwargs
    """

    _LONG_IDX  = [0, 1, 2, 3, 4, 5, 6, 9]  # OHLCV + bar_return + spread_pressure
    _SHORT_IDX = [6, 7, 8]                   # bar_return + wick_asym + vol_zscore

    def __init__(
        self,
        input_dim:   int   = 10,
        d_model:     int   = 512,
        n_heads:     int   = 8,
        ffn_dim:     int   = 2048,
        n_layers:    int   = 4,
        scatter_J:   int   = 3,
        scatter_Q:   int   = 4,
        filter_len:  int   = 31,
        attn_window: int   = 20,
        dropout:     float = 0.1,
        # Legacy kwargs — absorbed
        inception_channels:  int   = None,
        n_inception_blocks:  int   = None,
        kernel_sizes:        list  = None,
        tcn_layers:          int   = None,
        tcn_kernel_size:     int   = None,
        price_dropout:       float = None,
        n_bypass:    int   = 0,    # Phase 6: extra features injected post-scatter
        **_kwargs,
    ):
        super().__init__()
        if price_dropout is not None and dropout == 0.1:
            dropout = price_dropout

        n_long  = len(self._LONG_IDX)   # 8
        n_short = len(self._SHORT_IDX)  # 3

        # ── Long stream: Scattering + Transformer ────────────────────────────
        self.scatter = LearnableScatteringBlock(
            in_channels = n_long,
            J           = scatter_J,
            Q           = scatter_Q,
            filter_len  = filter_len,
            pool_size   = 2,
            dropout     = dropout,
        )
        scatter_out_ch = self.scatter.out_channels  # 8 + 8×12 = 104

        # Project scatter output → d_model
        self.scatter_proj = nn.Linear(scatter_out_ch, d_model)

        # Compute T_s: input T=240, pool=2 → T_s=120
        # Hardcode T_s=120 for static causal mask in TransformerEncoderLayer
        _T_s = 120
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model  = d_model,
                n_heads  = n_heads,
                ffn_dim  = ffn_dim,
                dropout  = dropout,
                seq_len  = _T_s,
            )
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Attention-weighted global pool
        self.long_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

        # ── Short stream: LocalCausalAttention(w=20) + AstrocyteGating ─────
        self.short_proj = nn.Linear(n_short, d_model)
        self.short_attn = nn.ModuleList([
            LocalCausalAttention(d_model, n_heads=4, window=attn_window,
                                 dropout=dropout)
            for _ in range(2)
        ])
        # Astrocyte routing replaces the hard pool — content-addressed retrieval
        # over K=16 learned pattern slots with regime-conditional temperature T
        self.astrocyte  = AstrocyteGatingModule(d_model, K=16, n_regimes=4)

        # ── Fusion ────────────────────────────────────────────────────────────
        # short_pooled is now (B, d_model) from astrocyte — no pool layer needed
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temperature scaling (post-training calibration — not trained with model)
        self.temp_scale = TemperatureScaling(init_T=1.5)

        # Phase 6 bypass: extra features (DXY, VWAP, session, MTF) skip scattering.
        # Scattering is designed for OHLCV microstructure — slow-moving exogenous
        # features alias badly through its filter bank. Project directly into
        # the d_model residual stream after attention pooling. n_bypass=0 → no-op.
        self.n_bypass = n_bypass
        if n_bypass > 0:
            self.bypass_proj = nn.Sequential(
                nn.Linear(n_bypass, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

        self.d_model        = d_model
        self.out_dim        = d_model
        self.attn_window    = attn_window
        self.use_checkpoint = True   # disabled at eval() automatically

    def forward(
        self,
        x:      torch.Tensor,          # (B, T=240, input_dim=10)
        regime: torch.Tensor | None = None,  # (B,) regime index 0-3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:      (B, T=240, input_dim=10)
            regime: (B,) integer regime index:
                    0=Bear+LOW, 1=Bear+HIGH, 2=Bull+LOW, 3=Bull+HIGH
                    None → uses default Bear+HIGH temperature

        Returns:
            pooled:       (B, d_model)
            seq_features: (B, T_s=120, d_model)
        """
        # ── Long stream ───────────────────────────────────────────────────────
        # First 10 dims only → scattering (bypass features skip filter bank)
        x_base   = x[:, :, :10] if x.shape[-1] > 10 else x
        x_long   = x_base[:, :, self._LONG_IDX]      # (B, 240, 8)
        scattered = self.scatter(x_long)              # (B, 120, 104)
        h        = self.scatter_proj(scattered)       # (B, 120, d_model)

        # Gradient checkpointing: recompute activations during backward
        # instead of storing them. Saves ~24 GB at B=4096 d=512 4-layer.
        # Cost: ~30% extra compute per backward pass — acceptable tradeoff.
        if self.training and self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            for layer in self.encoder:
                h = checkpoint(layer, h, use_reentrant=False)
        else:
            for layer in self.encoder:
                h = layer(h)
        h = self.encoder_norm(h)                      # (B, 120, d_model)

        long_scores  = self.long_pool(h).squeeze(-1)
        long_weights = torch.softmax(long_scores, dim=-1)
        long_pooled  = (h * long_weights.unsqueeze(-1)).sum(1)  # (B, d_model)

        # ── Short stream: LocalCausalAttention → AstrocyteGating ────────────
        x_short  = x_base[:, -self.attn_window:, self._SHORT_IDX]  # last 20 bars, short dims
        h_short  = self.short_proj(x_short)
        for layer in self.short_attn:
            h_short = layer(h_short)

        # Astrocyte routing: content-addressed pool with regime-conditional T
        # Replaces the fixed softmax-pool — gains concentrate on relevant patterns
        short_pooled = self.astrocyte(h_short, regime)  # (B, d_model)

        # ── Phase 6 bypass injection ─────────────────────────────────────────
        # Inject exogenous features (DXY, VWAP, session, MTF) into long_pooled
        # residual stream. Last bar's bypass dims are sufficient — these are
        # slow-moving signals representing current regime context, not sequence.
        if self.n_bypass > 0 and x.shape[-1] > 10:
            x_bp        = x[:, -1, 10:10 + self.n_bypass]  # (B, n_bypass)
            long_pooled = long_pooled + self.bypass_proj(x_bp)

        # ── Fusion ────────────────────────────────────────────────────────────
        pooled       = self.fusion(torch.cat([long_pooled, short_pooled], dim=-1))
        seq_features = h  # (B, 120, d_model) for cross-attention

        return pooled, seq_features


# ── Backward-compatible alias ─────────────────────────────────────────────────

class PriceBranch(TransformerPriceBranch):
    """Alias — fusion.py import unchanged."""
    pass