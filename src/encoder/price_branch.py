"""Multi-scale CNN/TCN price branch with InceptionTime-style residual blocks.

Processes OHLCV+spread tick data through parallel convolutions at multiple
scales (3/5/7/11 bar kernels), capturing microstructure patterns, swing
patterns, and trend structure simultaneously.

Architecture:
    Input (batch, seq, 6) → InceptionBlock × N → ResidualTCN → LayerNorm → output (batch, d_model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """InceptionTime-style multi-scale convolution block.

    Runs 4 parallel conv branches at different kernel sizes, concatenates,
    then projects down. Captures patterns at 3-bar, 5-bar, 7-bar, and
    11-bar horizons simultaneously.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_sizes: list[int] = [3, 5, 7, 11],
        dropout: float = 0.2,
    ):
        super().__init__()
        n_branches = len(kernel_sizes)
        branch_ch = max(out_channels // n_branches, 16)

        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, branch_ch, kernel_size=ks, padding=ks // 2, bias=False),
                nn.BatchNorm1d(branch_ch),
                nn.GELU(),
            ))

        # Max-pool branch for downsampled view
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.GELU(),
        )

        total_ch = branch_ch * (n_branches + 1)  # conv branches + pool branch

        # 1x1 projection to target dimension
        self.projection = nn.Sequential(
            nn.Conv1d(total_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len) → (batch, out_channels, seq_len)"""
        branch_outputs = [branch(x) for branch in self.branches]
        branch_outputs.append(self.pool_branch(x))
        concat = torch.cat(branch_outputs, dim=1)
        out = self.projection(concat)
        return out + self.residual(x)


class CausalTCNBlock(nn.Module):
    """Causal temporal convolution block with dilation and residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal: left-pad only

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding, bias=False
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding, bias=False
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.causal_trim = padding  # trim right side to enforce causality

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len) → (batch, channels, seq_len)"""
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        if self.causal_trim > 0:
            out = out[:, :, :-self.causal_trim]
        out = self.dropout(out)
        out = self.act(self.norm2(self.conv2(out)))
        if self.causal_trim > 0:
            out = out[:, :, :-self.causal_trim]
        out = self.dropout(out)
        return out + residual


class PriceBranch(nn.Module):
    """Multi-scale CNN/TCN encoder for OHLCV+spread price data.

    Architecture:
        Input → InceptionBlock (multi-scale) → InceptionBlock (deeper)
        → CausalTCN stack (temporal with exponential dilation)
        → LayerNorm → Global attention pooling → output

    The InceptionTime blocks extract local patterns at multiple scales,
    then the TCN stack models longer-range causal dependencies with
    exponentially increasing receptive field.
    """

    def __init__(
        self,
        input_dim: int = 6,
        inception_channels: int = 128,
        n_inception_blocks: int = 2,
        kernel_sizes: list[int] = [3, 5, 7, 11],
        tcn_layers: int = 4,
        tcn_kernel_size: int = 3,
        dropout: float = 0.2,
        d_model: int = 192,
    ):
        super().__init__()
        self.d_model = d_model

        # Multi-scale inception blocks
        inception = []
        in_ch = input_dim
        for i in range(n_inception_blocks):
            inception.append(InceptionBlock(in_ch, inception_channels, kernel_sizes, dropout))
            in_ch = inception_channels
        self.inception = nn.Sequential(*inception)

        # Project to d_model if needed
        self.channel_proj = (
            nn.Conv1d(inception_channels, d_model, kernel_size=1, bias=False)
            if inception_channels != d_model
            else nn.Identity()
        )

        # Causal TCN with exponential dilation
        self.tcn = nn.Sequential(*[
            CausalTCNBlock(d_model, tcn_kernel_size, dilation=2**i, dropout=dropout)
            for i in range(tcn_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Attention pooling: learn which timesteps matter most
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) — OHLCV+spread

        Returns:
            pooled: (batch, d_model) — global representation
            seq_features: (batch, seq_len, d_model) — per-timestep features (for cross-attention)
        """
        # Conv layers expect (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Multi-scale feature extraction
        x = self.inception(x)
        x = self.channel_proj(x)

        # Causal temporal modeling
        x = self.tcn(x)

        # Back to (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)
        seq_features = self.norm(x)

        # Attention-weighted pooling
        attn_scores = self.attn_pool(seq_features).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = (seq_features * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, d_model)

        return pooled, seq_features

    def receptive_field(self) -> int:
        """Calculate effective receptive field of the TCN stack."""
        tcn_layers = len(self.tcn)
        ks = 3
        rf = 1
        for i in range(tcn_layers):
            rf += 2 * (ks - 1) * (2 ** i)
        return rf
