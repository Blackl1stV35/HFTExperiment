"""Cross-attention fusion + confidence head — the full dual-branch model.

Fuses price sequence features with sentiment context via cross-attention,
then produces both a 3-class signal (buy/hold/sell) and a scalar confidence
score that feeds into position sizing.

Architecture:
    PriceBranch (seq_features) × SentimentBranch (context)
    → Cross-Attention: price queries, sentiment keys/values
    → Early fusion: concat mid-layer features
    → Late fusion: separate signal + confidence heads merged at output
    → Output: logits (batch, 3) + confidence (batch, 1)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder.price_branch import PriceBranch
from src.encoder.sentiment_branch import SentimentBranch


class CrossAttentionFusion(nn.Module):
    """Cross-attention: price features attend to sentiment context.

    Price sequence features (queries) attend to sentiment embedding (keys/values).
    This lets the model learn HOW sentiment should modulate price interpretation —
    e.g., a doji candle after hawkish Fed news should be interpreted differently
    than the same doji during quiet markets.
    """

    def __init__(self, d_model: int = 192, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        price_features: torch.Tensor,
        sentiment_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            price_features: (batch, seq_len, d_model) — from PriceBranch
            sentiment_context: (batch, d_model) — from SentimentBranch

        Returns:
            (batch, seq_len, d_model) — sentiment-modulated price features
        """
        # Expand sentiment to (batch, 1, d_model) for cross-attention K/V
        sent_kv = sentiment_context.unsqueeze(1)  # (batch, 1, d_model)

        # Cross-attention: price queries attend to sentiment
        attn_out, _ = self.cross_attn(
            query=price_features,
            key=sent_kv,
            value=sent_kv,
        )
        x = self.norm(attn_out + price_features)

        # Feed-forward
        x = self.norm2(self.ffn(x) + x)
        return x


class DualBranchModel(nn.Module):
    """Full dual-branch model: Price CNN/TCN + FinBERT sentiment + fusion.

    The model takes two inputs:
        1. Price sequence (batch, seq_len, 6) — OHLCV+spread
        2. Sentiment embedding (batch, 768) — FinBERT daily consensus

    And produces two outputs:
        1. Signal logits (batch, 3) — buy/hold/sell classification
        2. Confidence (batch, 1) — scalar confidence for position sizing

    Architecture (SentiStack pattern):
        Early fusion: concat price_pooled + sentiment_encoded at mid-layer
        Late fusion: separate heads for signal + confidence, merged at decision

    The confidence head is trained with the signal head but produces an
    independent scalar — high confidence = strong directional conviction,
    low confidence = uncertain/hold-leaning.
    """

    def __init__(
        self,
        # Price branch
        input_dim: int = 6,
        inception_channels: int = 128,
        n_inception_blocks: int = 2,
        kernel_sizes: list[int] = [3, 5, 7, 11],
        tcn_layers: int = 4,
        tcn_kernel_size: int = 3,
        price_dropout: float = 0.2,
        d_model: int = 192,
        # Sentiment branch
        sentiment_input_dim: int = 768,
        sentiment_hidden_dim: int = 256,
        sentiment_dropout: float = 0.3,
        # Fusion
        fusion_heads: int = 4,
        fusion_dropout: float = 0.1,
        # Classifier
        classifier_dims: list[int] = [128, 64],
        classifier_dropout: float = 0.4,
        n_classes: int = 3,
    ):
        super().__init__()

        # Price branch: multi-scale CNN/TCN
        self.price_branch = PriceBranch(
            input_dim=input_dim,
            inception_channels=inception_channels,
            n_inception_blocks=n_inception_blocks,
            kernel_sizes=kernel_sizes,
            tcn_layers=tcn_layers,
            tcn_kernel_size=tcn_kernel_size,
            dropout=price_dropout,
            d_model=d_model,
        )

        # Sentiment branch: FinBERT → MLP
        self.sentiment_branch = SentimentBranch(
            input_dim=sentiment_input_dim,
            hidden_dim=sentiment_hidden_dim,
            d_model=d_model,
            dropout=sentiment_dropout,
        )

        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(d_model, fusion_heads, fusion_dropout)

        # Attention pooling for fused features
        self.fused_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

        # Early fusion: concat price + sentiment pooled representations
        fused_dim = d_model * 2  # price_pooled + sentiment_encoded concatenated

        # Signal head (buy/hold/sell)
        signal_layers = []
        dim_in = fused_dim
        for dim_out in classifier_dims:
            signal_layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
            ])
            dim_in = dim_out
        signal_layers.append(nn.Linear(dim_in, n_classes))
        self.signal_head = nn.Sequential(*signal_layers)

        # Confidence head (scalar 0-1)
        self.confidence_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.d_model = d_model
        self.n_classes = n_classes
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        price_seq: torch.Tensor,
        sentiment: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            price_seq: (batch, seq_len, input_dim) — OHLCV+spread
            sentiment: (batch, 768) — FinBERT embedding (optional)

        Returns:
            signal_logits: (batch, n_classes) — buy/hold/sell
            confidence: (batch, 1) — confidence score [0, 1]
        """
        # Price branch: get both pooled and sequence-level features
        price_pooled, price_seq_features = self.price_branch(price_seq)

        # Sentiment branch
        if sentiment is None:
            sentiment = torch.zeros(price_seq.shape[0], 768, device=price_seq.device)
        sent_encoded = self.sentiment_branch(sentiment)  # (batch, d_model)

        # Cross-attention: price attends to sentiment
        fused_seq = self.cross_attention(price_seq_features, sent_encoded)

        # Pool the cross-attended features
        attn_scores = self.fused_pool(fused_seq).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        fused_pooled = (fused_seq * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Early fusion: concat price + fused representations
        combined = torch.cat([price_pooled, fused_pooled], dim=-1)  # (batch, d_model*2)

        # Signal head
        signal_logits = self.signal_head(combined)

        # Confidence head
        confidence = self.confidence_head(combined)

        return signal_logits, confidence

    def predict(self, price_seq: torch.Tensor, sentiment: Optional[torch.Tensor] = None) -> dict:
        """Convenience method for inference — returns action, confidence, probabilities."""
        self.eval()
        with torch.no_grad():
            logits, confidence = self.forward(price_seq, sentiment)
            probs = F.softmax(logits, dim=-1)
            action = probs.argmax(dim=-1)

        return {
            "action": action,          # (batch,) — 0=sell, 1=hold, 2=buy
            "confidence": confidence,   # (batch, 1) — [0, 1]
            "probabilities": probs,     # (batch, 3)
            "logits": logits,           # (batch, 3)
        }

    @classmethod
    def from_config(cls, cfg) -> "DualBranchModel":
        """Build from Hydra config."""
        return cls(
            input_dim=cfg.input.feature_dim,
            inception_channels=cfg.price.inception_channels,
            n_inception_blocks=cfg.price.n_inception_blocks,
            kernel_sizes=cfg.price.kernel_sizes,
            tcn_layers=cfg.price.tcn_layers,
            tcn_kernel_size=cfg.price.tcn_kernel_size,
            price_dropout=cfg.price.dropout,
            d_model=cfg.d_model,
            sentiment_input_dim=cfg.sentiment.input_dim,
            sentiment_hidden_dim=cfg.sentiment.hidden_dim,
            sentiment_dropout=cfg.sentiment.dropout,
            fusion_heads=cfg.fusion.heads,
            fusion_dropout=cfg.fusion.dropout,
            classifier_dims=cfg.classifier.hidden_dims,
            classifier_dropout=cfg.classifier.dropout,
            n_classes=cfg.classifier.n_classes,
        )
