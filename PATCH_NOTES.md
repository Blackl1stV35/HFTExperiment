# HFTExperiment — Patch Notes

## Phase 3 — Regime-Informed Pipeline (current branch)

### Overview
Full integration of the market regime research (v5 notebook, 25 cells, 2005–2026)
into the training and RL pipeline. The research phase is complete. All six pipeline
steps are implemented in this branch.

### Research outputs consumed
- `data/regime/daily_regime_labels.csv` — 7,344 daily rows, 35 columns
  - `gmm2_state`         — 2-state GMM (Bear=0 / Bull=1), 5-day min-dwell smoothed
  - `km_label_63d`       — KMeans cluster sorted by Sharpe, 63d-EMA smoothed
  - `vol_regime`         — XAUUSD annualised vol (LOW/NORMAL/HIGH, thresholds 10.5/14.3%)
  - `regime_quality_norm`— [0,1] Sharpe scalar from GMM×vol heatmap (obs[10])
  - `gs_quartile_enc`    — G/S ratio quartile rank [0,1] (Q1=silver-leads, Q4=gold-leads)
  - `cu_au_regime_enc`   — Copper-gold correlation regime (Financial/Mixed/Commodity)
  - `dfii10_real_yield`  — FRED TIPS 10yr real yield

### Key research findings driving each change
| Finding | Change |
|---------|--------|
| GMM Bear Sharpe ≈ 0 (2-state Bear = −0.44) — Bear = "not Bull" | GMM2 Bull flag as sole entry gate (Step 6) |
| G/S Q1 optimal hold = 7 days daily; Q4 = 80 days | G/S-conditioned max_hold: Q1→40, Q2/3→60, Q4→80 bars (Step 5) |
| Bear+HIGH vol = Sharpe 0.33 (worst reliable cell) | regime_quality_norm scalar in obs (Step 4) |
| Cu-Au 0.5 threshold confirmed optimal; Commodity regime Sharpe 0.63 | cu_au_regime_enc in obs (Step 4) |
| 78% GMM Bear in 2026 / 0% Bull — training coverage gap | Regime-balanced sampler: Bear ×2.0, pre-2020 ×1.5, post-2024 ×0.5 (Step 3) |
| Transition matrix: P(exit Bear today) = 0.7%, 50/50 crossover at ~42d | Deploy at reduced size until vol drops below 14.3% |

### Files changed

#### New / replaced
| File | Change |
|------|--------|
| `src/data/preprocessing.py` | Added `join_regime_labels()`, `get_regime_array()`, encoding maps |
| `src/training/train_supervised.py` | Added `build_regime_balanced_sampler()`, regime-balanced DataLoader |
| `scripts/train_rl.py` | Obs 10→13, G/S max_hold conditioning, GMM2 entry gate |
| `configs/data/xauusd.yaml` | Added `class_weights: [2.5, 0.3, 2.5]`, confirmed `max_holding_bars: 80` |
| `configs/rl/sac.yaml` | `obs_dim: 13`, Phase 3 regime section, G/S hold limits |
| `notebooks/00_market_regime_explorer_v5.ipynb` | Final 25-cell research notebook |

#### Unchanged from Phase 2
- `src/meta_policy/rl_agent.py` — ConfidenceSACAgent (obs_dim now passed as 13)
- `src/data/feature_engineering.py` — compute_rl_obs_features()
- `src/encoder/fusion.py` — DualBranchModel (frozen)
- `src/backtesting/engine.py`
- `src/hitl/mt5_interface.py`
- `scripts/download_data.py`

### Step-by-step execution (new branch)

```bash
# Step 1: Download 6666-day M1 XAUUSD (prerequisite for all ML work)
python scripts/download_data.py --source mt5 --symbol XAUUSD --timeframe M1 --days 6666

# Run research notebook to generate daily_regime_labels.csv
# (or copy existing CSV from prior run)
# jupyter notebook notebooks/00_market_regime_explorer_v5.ipynb

# Step 2+3: Supervised training (regime-balanced sampler auto-activates if CSV present)
python scripts/train_supervised.py model=dual_branch data=xauusd

# Steps 4+5+6: RL training (obs=13, G/S hold, GMM2 gate)
python scripts/train_rl.py \
    --checkpoint models/dual_branch_best.pt \
    --regime-csv data/regime/daily_regime_labels.csv \
    --steps 500000 --seed 42 \
    --mtm-scale 0.05 --hold-penalty 0.003 --early-cut-bonus 0.40 \
    --curriculum-warmup 100000
```

### Observation vector (13-dim)
```
Index  Feature                 Source          Notes
0      sell_prob               supervised      softmax[0]
1      hold_prob               supervised      softmax[1]
2      buy_prob                supervised      softmax[2]
3      confidence              supervised      confidence head [0,1]
4      position_dir            env             -1/0/1
5      unrealized_pnl_norm     env             clipped [-5,5]
6      hold_time_norm          env             hold/max_hold [0,1]
7      atr_norm                feature_eng     rolling ATR/close [0,0.05]
8      trend_norm              feature_eng     EMA slope clipped [-2,2]
9      session_phase           feature_eng     London/NY [0,1]
10     regime_quality_norm     regime_csv      GMM×vol heatmap Sharpe [0,1]  ← NEW
11     gs_quartile_norm        regime_csv      G/S rank [0=silver-leads, 1=gold-leads] ← NEW
12     cu_au_regime_enc        regime_csv      Cu-Au corr [0=Financial, 1=Commodity]   ← NEW
```

### Current macro context (as of 2026-04-17)
- GMM2: **Bear** (P(exit today)=0.7%, 50/50 crossover at ~42 days forward)
- KMeans 63d: **Regime-B** (Sharpe 0.66)
- Vol regime: **HIGH** (19.9% ann — worst reliable cell with Bear = Sharpe 0.33)
- G/S quartile: **Q1** (rank 0.15 — silver outperforming → max_hold=40 bars)
- Cu-Au regime: **Commodity** (252d corr=0.52, Sharpe 0.63)
- Business cycle: **Stagflation Q3** (growth z=−0.27, inflation z=+2.34)
- DFII10: **+1.93%** — real yield framework broken post-2022 ($4,057 unexplained premium)
- Deploy: reduced size until vol crosses below 14.3% HIGH threshold

---

## Phase 2 — SAC v1–v4 (prior branch)

### v4 (final Phase 2)
- RL obs 7→10: added atr_norm, trend_norm, session_phase
- compute_rl_obs_features() in feature_engineering.py
- Class weights [2.5, 0.3, 2.5] in xauusd.yaml
- download_data.py: chunked MT5 (30d windows, 0.5s sleep) + yfinance fallback

### v3
- Curriculum EXIT_THRESHOLD 0.0→−0.10 at 100k steps
- Removed pre-trade confidence gate from env (reward gate only)

### v2
- max_hold aligned to 80 bars (sweep-optimal)
- MTM step reward (0.05×), quadratic hold penalty (0.003×)
- Early-cut bonus (0.40×), dual-output actor [pos_size, exit_logit]

### v1
- Initial SAC agent with ConfidenceSACAgent
- Triple barrier labels, 3-class supervised model
