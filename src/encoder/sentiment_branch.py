"""FinBERT sentiment branch: encodes news embeddings into trading-relevant features.

Takes pre-computed FinBERT embeddings (768-dim) and transforms them through
a small MLP into a compact sentiment representation that can be fused with
price features via cross-attention.

For training: uses pre-built daily consensus embeddings from data/sentiment_embeddings.npy
For inference: uses real-time GDELT → FinBERT pipeline
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SentimentBranch(nn.Module):
    """MLP encoder for FinBERT sentiment embeddings.

    Projects the 768-dim FinBERT hidden state into a compact d_model-dim
    representation suitable for cross-attention fusion with price features.

    Architecture:
        FinBERT 768 → Linear → GELU → LayerNorm → Dropout
        → Linear → GELU → LayerNorm → Dropout → output (d_model)

    The output is a single vector per sample (not a sequence), which gets
    expanded and cross-attended against the price sequence in the fusion layer.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        d_model: int = 192,
        dropout: float = 0.3,
        n_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        layers = []
        dim_in = input_dim
        for i in range(n_layers):
            dim_out = hidden_dim if i < n_layers - 1 else d_model
            layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.GELU(),
                nn.LayerNorm(dim_out),
                nn.Dropout(dropout),
            ])
            dim_in = dim_out

        self.encoder = nn.Sequential(*layers)

        # Learnable "no-sentiment" embedding for bars with zero sentiment
        self.null_embedding = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 768) — FinBERT embedding (may be all-zeros if no news)

        Returns:
            (batch, d_model) — encoded sentiment representation
        """
        # Detect zero embeddings (no news available)
        is_null = (x.abs().sum(dim=-1) < 1e-6)  # (batch,)

        encoded = self.encoder(x)  # (batch, d_model)

        # Replace zero-input outputs with learnable null embedding
        if is_null.any():
            encoded[is_null] = self.null_embedding.unsqueeze(0).expand(is_null.sum(), -1)

        return encoded
