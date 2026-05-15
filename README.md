# Ongoing Experiment (update: 15/05/2026)

I am open for contribution, if you are interested in applying machine processing into financial time-series forecast, please contact me at kanokphan.s@ku.th. I am eager for any discussions toward investment research and trading. 

### To-do Lists
- [ ] Re-labelling distribution approach
- [ ] Collecting paper trade logs for direction change 

(see: [Release Note v2.13.9: Technical Research Plan](https://github.com/Blackl1stV35/HFTExperiment/releases/tag/v2.13.9))

## Phase 5 Core Architecture

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
LearnableScatteringBlock(J=3, Q=4)
    → (B, 120, 104)
    ↓
scatter_proj: Linear(104, 512)
    → (B, 120, 512)
    ↓
TransformerEncoder(
    d_model=512, n_heads=8, ffn_dim=2048,
    n_layers=4, dropout=0.1,
    is_causal=True
)
    → (B, 120, 512)
    ↓
Attention-weighted Pool
    → z_main (B, 512)

────────────────────────────────────────

Short Stream (wick_asym, vol_z) (B, 240, 2)
    ↓
LocalCausalAttention(w=20)
    → (B, 240, 512)
    ↓
Last-bar state extraction
    → x_last (B, 512)

    ↓
    Regime detection (from sentiment tensor)
        → regime ∈ {0,1,2,3}
        → T_regime (learned, log-space)

    ↓
AstrocyteRoutingModule(K=16, conditioned on T_regime)
    ├─ Pattern Memory ξ₁…ξ₁₆ ∈ ℝ^512
    ├─ Pattern fitness:
         fμ = ⟨x_last, ξμ⟩ / D
    ├─ Gains:
         pμ = softmax(fμ / T_regime)
    ├─ Retrieval:
         r = Σμ pμ ξμ
    └─ Residual blend:
         z_short = g · r + (1 − g) · x_last

    → z_short (B, 512)

────────────────────────────────────────

Fusion
    Concatenate z_main + z_short → (B, 1024)
    ↓
    Linear(1024 → 512)
    ↓
    LayerNorm
    ↓
    GELU
    → z_fused (B, 512)
    ↓
Logits → (B, C)

    ↓
TemperatureScaling (post-hoc, scalar T)
    ↓
Calibrated Probabilities (B, C)
```

## Core Project Structure

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
