# XAUUSD Neural Network Futures Trading System v2

Consensus-aligned deep learning trading system for XAUUSD (Gold) futures.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DUAL-BRANCH ENCODER                                        │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │ Price Branch      │    │ Sentiment Branch  │              │
│  │ Multi-scale CNN/  │    │ FinBERT → MLP     │              │
│  │ TCN + Residual    │    │                   │              │
│  └────────┬─────────┘    └────────┬──────────┘              │
│           └──────────┬───────────┘                          │
│              Cross-Attention Fusion                         │
│              + Confidence Head                              │
├─────────────────────────────────────────────────────────────┤
│  META-POLICY / HIERARCHICAL RL                              │
│  Regime Router → Expert Agents (SAC)                        │
│  GAN-Simulated Market for RL Training                       │
│  Confidence → Position Sizing                               │
├─────────────────────────────────────────────────────────────┤
│  HITL EXECUTION                                             │
│  MT5 Interface → Risk Display → Human Approval Gate         │
│  Circuit Breakers → Order Execution                         │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

- **CNN/TCN over Mamba SSM**: Multi-scale convolutions handle M1 tick data better at 120-240 bar sequences. Numerically stable, 4x faster training.
- **Dual-branch fusion**: Price features and sentiment features are processed independently then fused via cross-attention — captures how news impacts price.
- **Confidence output**: Alongside buy/hold/sell, the model outputs a confidence score that feeds into position sizing.
- **HITL exits**: Human operator approves/vetoes exits, especially on large positions or drawdown events.
- **GAN market simulation**: Generates synthetic market scenarios for RL training to improve generalization.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py --synthetic --days 90
python scripts/train_supervised.py model=dual_branch data=xauusd
python scripts/backtest.py model=dual_branch data=xauusd
```

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
