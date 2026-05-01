## Underdevelopment 

## Phase 5 Architecture 

Digram Sketch: v0.5.3.1-PriceBranchTransformer-ScatterPool-enriched
- (1) Astrocyte Routing: K=16 learned pattern slots
- (2) Regime-Conditional Temperature ($T$)
- (3) Temperature Scaling: Initializing $T=1.5$ and using L-BFGS for post-training fit.

<p align="center">
  <img src="assets\v0.5.3.1-PriceBranchTransformer-ScatterPool-enriched.png" alt="Phase 5.3 Architecture Diagram" style="max-width:100%; height:auto;"/>
</p>

```
Input (B, 240, 10)
    ↓
LearnableScatteringBlock(J=3, Q=4)    ← KEEP: validated discriminative signal
    → (B, 120, 104)
    ↓
scatter_proj: Linear(104, 512)
    → (B, 120, 512)
    ↓
TransformerEncoder(
    d_model=512, n_heads=8, ffn_dim=2048,
    n_layers=4, dropout=0.1,
    is_causal=True                    ← causal mask, static shapes ✓
)   → (B, 120, 512)
    ↓
Attention-weighted pool → (B, 512)

Short stream (wick_asym, vol_z): unchanged LocalCausalAttention(w=20)
    ↓
Fusion: Linear(512+512, 512) → LayerNorm → GELU
```

## Project Structure (not finalised)

```
HFTExperiment/
├── configs/              # Hydra configuration
├── src/
│   ├── encoder/          # Dual-branch encoder architecture
│   │   ├── price_branch.py      # Multi-scale CNN/TCN + residual
│   │   ├── sentiment_branch.py  # FinBERT → MLP encoder
│   │   └── fusion.py            # Cross-attention + confidence head
│   ├── meta_policy/      # Hierarchical RL
│   │   ├── rl_agent.py          # SAC with confidence-based sizing
│   │   └── gan_market.py        # GAN-simulated market
│   ├── hitl/             # Human-in-the-loop
│   │   ├── mt5_interface.py     # MT5 signal + approval gate
│   │   └── risk_display.py      # Explainable feature readout
│   ├── training/         # Training pipelines
│   │   ├── labels.py            # 3-class labeling
│   │   ├── train_supervised.py  # Supervised training loop
│   │   └── train_rl.py          # RL training loop
│   ├── data/             # Data ingestion & preprocessing
│   ├── backtesting/      # Backtesting engine
│   ├── risk/             # Circuit breakers & uncertainty
│   ├── execution/        # Broker integration
│   ├── inference/        # ONNX export & inference
│   ├── monitoring/       # Prometheus + Telegram
│   └── utils/            # Shared utilities
├── scripts/              # Entry point scripts
├── tests/                # Tests
└── rust_inference/       # Optional Rust inference engine
```

## License
This code is proprietary and confidential. Unauthorized copying, distribution, or use of this file, via any medium, is strictly prohibited
