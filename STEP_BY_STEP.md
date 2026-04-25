# Step-by-Step Guide — HFTExperiment v2

Follow in order. Each step has a prerequisite. Do not skip ahead.

---

## Prerequisites

```bash
# Mount Google Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/Colab Notebooks", exist_ok=True)

# Clone / upload the project
# Upload HFTExperiment-v2.zip to Colab, then:
!unzip HFTExperiment-v2.zip -d /content/
cd /content/HFTExperiment-v2

# Install dependencies
pip install -r requirements.txt --break-system-packages
```

---

## Step 1 — Download M1 data

**Prerequisite:** MT5 credentials in `.env` (copy from `.env.example`)

```bash
python scripts/download_data.py \
    --source mt5 --symbol XAUUSD --timeframe M1 --days 6666
```

Produces: `data/ticks.duckdb` (~2 GB)

---

## Step 2 — Generate regime labels

Run the research notebook to produce `data/regime/daily_regime_labels.csv`:

```bash
jupyter notebook notebooks/00_market_regime_explorer_v5.ipynb
```

Or copy an existing CSV directly:
```bash
cp "/content/drive/MyDrive/Colab Notebooks/daily_regime_labels.csv" \
   data/regime/daily_regime_labels.csv
```

The CSV is required for the regime-balanced sampler and RL regime conditioning.
If absent, `join_regime_labels()` falls back to neutral defaults and logs a warning.

---

## Step 3 — Supervised training

**Prerequisite:** Steps 1-2 complete.

Update `configs/config.yaml` before running:
```yaml
training:
  learning_rate: 5e-5
  gradient_clip: 10.0
  focal_gamma: 1.0
  early_stopping_patience: 15

# configs/data/xauusd.yaml
labeling:
  class_weights: [2.5, 0.3, 5.0]
```

Run from scratch (recommended):
```bash
python scripts/train_supervised.py \
    model=dual_branch data=xauusd \
    training.batch_size=2048 training.num_workers=0
```

Resume from Drive checkpoint:
```bash
python scripts/train_supervised.py \
    model=dual_branch data=xauusd \
    training.batch_size=2048 training.num_workers=0 \
    +training.resume_from="/content/drive/MyDrive/Colab Notebooks/dual_branch_best.pt"
```

**Target metrics (accept checkpoint when):**
- Test sell F1 ≥ 0.30
- Test sell precision ≥ 0.25
- Val signal score (F1-based) ≥ 0.025

**Current best:** ep60, sell P=0.253 R=0.611 F1=0.358 (use this as baseline)

---

## Step 4 — Confidence calibration (v2 improvement, §5.1)

After supervised training completes, calibrate the confidence head against
realised Sharpe on the validation split.

```python
# In a Colab cell after training:
from src.training.confidence_calibration import SharpeConfidenceHead, IsotonicCalibrator
import numpy as np

# Load val confidences + close prices (from your DataLoader)
# raw_conf_val: (n_val,) array of raw confidence outputs
# close_prices_val: (n_val,) close prices
# val_seq_indices: array of sequence end indices

realised = SharpeConfidenceHead.compute_realised_sharpe(
    close_prices_val, val_seq_indices, n_bars=20
)
cal = IsotonicCalibrator()
cal.fit(raw_conf_val, realised)
cal.save("models/confidence_calibrator.pkl")

# Test: calibrated confidence should correlate with future returns
calibrated = cal.transform(raw_conf_val)
print(f"Calibrated conf mean: {calibrated.mean():.3f}  std: {calibrated.std():.3f}")
```

Update the RL `confidence_gate` to use calibrated confidence at inference.

---

## Step 5 — RL training with cost curriculum

**Prerequisite:** Step 3 complete. Checkpoint at `models/dual_branch_best.pt`.

```bash
python scripts/train_rl.py \
    --checkpoint models/dual_branch_best.pt \
    --regime-csv data/regime/daily_regime_labels.csv \
    --steps 1500000 --n-evolves 10 \
    --episode-len 8000 \
    --eval-every 16000 --n-eval-episodes 15 \
    --confidence-gate 0.70 \
    --commission 0.70 --spread-pips 2.0 \
    --mtm-scale 0.05 --hold-penalty 0.003 --early-cut-bonus 0.40 \
    --curriculum-warmup 100000 \
    --seed 42
```

**What to watch:**
- `evolve=0` (steps 0-150k): commission=0, spread=0. Agent should achieve positive eval PnL.
  If it cannot win in a frictionless environment, the supervised signal is insufficient.
- `evolve=5` (steps 750k): commission=$0.35, spread=1.0 pip. Expect some PnL reduction.
- `evolve=9` (steps 1.35M): full costs. Eval PnL should still be positive if edge is real.
- WR should exceed 50% by evolve 3-4 in the frictionless window.

**Checkpoints saved:**
- `models/rl_agent_evolve{0-9}.pt` — one per evolve stage
- `models/rl_agent_best.pt` — best eval across all steps

**Deployment routing by regime:**
```python
# In paper_trade.py / backtest.py:
if vol_regime == "HIGH" and gmm2 == "Bear":
    agent_path = "models/rl_agent_evolve9.pt"   # conservative, high cost
elif vol_regime == "LOW" and gmm2 == "Bull":
    agent_path = "models/rl_agent_evolve3.pt"   # active, low effective cost
else:
    agent_path = "models/rl_agent_best.pt"
```

---

## Step 6 — Backtest

```bash
python scripts/backtest.py \
    --supervised-checkpoint models/dual_branch_best.pt \
    --rl-checkpoint models/rl_agent_best.pt \
    --start 2024-01-01 --end 2026-04-01
```

---

## Step 7 — A/B test Sessa vs CCSO encoder (optional, §1.1)

To compare the Sessa mixer against the default CCSO two-stage branch:

```bash
# Run supervised training with Sessa variant
# (implement SessaLayer.forward() in src/encoder/price_branch_sessa.py first)
python scripts/train_supervised.py \
    model=dual_branch_sessa data=xauusd \
    training.batch_size=2048 training.num_workers=0
```

See `src/encoder/price_branch_sessa.py` for the TODO list:
implement `torch.linalg.solve_triangular` forward path.

---

## Step 8 — LeWM world model (optional, §4.2)

Train the JEPA latent world model as a complement/replacement for GAN market sim:

```python
from src.meta_policy.lewm_world import LeWMWorldModel

world = LeWMWorldModel(latent_dim=256, n_projections=256, sigreg_lambda=0.1)
# Training loop:
for batch in latent_loader:
    z_t, z_t1, actions = batch
    z_pred = world.predict(z_t, actions)
    loss = world.loss(z_pred, z_t1, z_t)
    loss.backward(); opt.step()

# Planning:
best_actions = world.plan_cem(z_current, horizon=5)
```

---

## Current macro context (2026-04-25)

| Signal | Value | Implication |
|--------|-------|-------------|
| GMM2 | Bear (P_exit=0.7%/day) | Block new entries per RL gate |
| Vol regime | HIGH (19.9% ann) | Use evolve 9 (cautious) checkpoint |
| G/S quartile | Q1 (rank 0.15) | max_hold=40 bars |
| Cu-Au regime | Commodity | Normal signal weighting |
| Bear+HIGH cell | Sharpe 0.33 | Worst reliable cell; reduce size |

**Recommendation:** wait for vol to cross below 14.3% (NORMAL threshold) before
deploying any evolve < 7 checkpoint.
