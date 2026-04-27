## Under-development 

## Project Structure

```
xauusd-v2/
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
