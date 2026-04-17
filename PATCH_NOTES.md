# Patch Notes — v2 (Phase 2 RL)

## Summary

Five targeted fixes addressing the R:R collapse observed in the Phase 2 backtest
results. The supervised signal (52.9% WR) is valid. The problem was entirely in
the execution and exit management layer of the RL agent.

**Root cause:** `max_position_time=120` in the backtest engine vs `max_hold=80`
in the RL training env, combined with no intermediate reward signal, meant the
agent was force-closed at 120 bars on nearly every losing trade — bloating avg
loss to $575 vs the label-designed 150 pip ($15) stop.

---

## Files changed

| File | Strategies applied |
|---|---|
| `src/meta_policy/rl_agent.py` | S3 |
| `scripts/train_rl.py` | S1, S2, S3, S4 |
| `src/backtesting/engine.py` | S1, S5a |
| `src/hitl/mt5_interface.py` | S5b |
| `configs/rl/sac.yaml` | S1, S2, S4 (params) |
| `configs/data/xauusd.yaml` | S1 (label horizon) |

---

## Strategy 1 — Horizon alignment + quadratic hold penalty

**Files:** `train_rl.py`, `engine.py`, `configs/rl/sac.yaml`, `configs/data/xauusd.yaml`

All three execution horizons now agree:
- `configs/data/xauusd.yaml` → `max_holding_bars: 80`
- `FrozenEncoderEnv` → `max_hold=80` default
- `BacktestConfig` → `max_position_time=80` default

Quadratic hold penalty added to `FrozenEncoderEnv.step()`:

```python
hold_frac = self.hold_time / max(self.max_hold, 1)
reward -= hold_penalty_coeff * (hold_frac ** 2)   # default: 0.002
```

The penalty is light at first (agent is free to hold winning trades) but
compounds strongly past 50% of max_hold, creating a gradient toward earlier
exits on losing positions.

---

## Strategy 2 — Mark-to-market step reward

**Files:** `train_rl.py`

Dense per-step reward added to `FrozenEncoderEnv.step()`:

```python
mtm_delta = current_unrealized_usd - prev_unrealized_usd
reward += mtm_delta * mtm_scale   # default: 0.05
```

The 0.05 scale keeps the MTM contribution small relative to the terminal
reward, so it shapes rather than dominates. The critic now receives gradient
signal at every bar about trade trajectory, enabling it to value voluntary
exits at -$200 as better than forced exits at -$575.

---

## Strategy 3 — Dedicated exit head on actor

**Files:** `rl_agent.py`, `train_rl.py`

Actor output extended from 1-dim to **2-dim**:

```
action[0]: position size  [-1, +1]  (scaled by confidence, as before)
action[1]: exit logit     [-1, +1]  (< EXIT_THRESHOLD = -0.10 → close)
```

Twin critics updated to accept 2-dim action (obs_dim + 2).

The exit head gives the agent an independent gradient path for closure
decisions. Under the old 1-dim design, the "go flat" signal was encoded as
`|action| < 0.15` — a dead zone with no gradient. The exit logit always
has gradient flowing through it.

**Migration note:** existing `rl_agent_best.pt` checkpoints are incompatible
(actor shape changed). Retrain from scratch.

---

## Strategy 4 — Asymmetric early-cut bonus

**Files:** `train_rl.py`

In `_close_position(voluntary=True)`, when the agent voluntarily closes
a losing trade before 80% of `max_hold`:

```python
remaining_bars = max_hold - hold_time
avg_pip_per_bar = abs(pnl_usd) / max(hold_time, 1)
avoided_loss = avg_pip_per_bar * remaining_bars
early_cut_bonus = min(avoided_loss * 0.30, abs(pnl_usd) * 0.50)
```

The bonus is estimated from the trade's own loss rate, capped at 50% of
actual loss to prevent gaming. This directly incentivises cutting at -$200
rather than riding to -$575.

---

## Strategy 5 — HITL as genuine alpha source

**Files:** `engine.py`, `mt5_interface.py`

**5a — Mid-hold review trigger (`engine.py`):**

`BacktestEngine.run()` now checks unrealized PnL every bar while in position.
When `unrealized_pnl <= hitl_mid_hold_threshold_usd` (default -$200), a
HITL review fires once per position (tracked via `_mid_hold_reviewed` set).
This surfaces cut decisions to the human before `max_hold` expiry.

Enable with `BacktestConfig(human_exit_approval=True, hitl_mid_hold_review=True)`.

**5b — DrawdownContext (`mt5_interface.py`):**

New `DrawdownContext` dataclass carries:
- `consecutive_losses` — current losing streak
- `daily_pnl_usd` — today's running PnL
- `session_volatility_pips` — ATR proxy
- `account_drawdown_pct` — derived from peak/current balance

`SignalContext` has a new optional `drawdown_ctx: DrawdownContext` field.
The console approval UI renders a DRAWDOWN CONTEXT block when present,
giving the human reviewer full situational awareness.

Auto-approve threshold raised from 0.70 → 0.72.
Profitable exits during consecutive-loss streaks (≥3) are now reviewed
rather than auto-approved — detects regime breaks before they compound.

---

## Recommended training invocation (post-patch)

```bash
python scripts/train_rl.py \
  --checkpoint models/dual_branch_best.pt \
  --max-hold 80 \
  --confidence-gate 0.48 \
  --eval-every 8000 \
  --seed 42 \
  --steps 500000 \
  --mtm-scale 0.05 \
  --hold-penalty 0.002 \
  --early-cut-bonus 0.30
```

## Recommended backtest invocation (post-patch)

```bash
# Standard (no HITL)
python scripts/backtest.py model=dual_branch data=xauusd \
  ++min_confidence=0.7

# With full HITL (mid-hold reviews enabled)
python scripts/backtest.py model=dual_branch data=xauusd \
  ++min_confidence=0.7 \
  ++risk.human_exit_approval=true
```

---

## Patch v3 — post-run-4 analysis

Applied after evaluating 4 runs from the v2 patch. Run 4 (no hard gate, avg
conf 0.636, 54 trades) achieved PF 0.98 and Sharpe -0.10 — closest to
breakeven, better than all gated runs including 0.70 and 0.83. This confirmed
three actionable changes.

### Files changed in v3

| File | Change |
|---|---|
| `src/meta_policy/rl_agent.py` | Curriculum EXIT_THRESHOLD |
| `scripts/train_rl.py` | Coefficients + curriculum call + gate removal |
| `src/backtesting/engine.py` | min_confidence gate removed from run() |
| `scripts/backtest.py` | min_confidence removed from BacktestConfig build |
| `configs/rl/sac.yaml` | Updated defaults |

### Change 1 — Remove pre-trade confidence gate

**Files:** `engine.py`, `backtest.py`, `train_rl.py`

`min_confidence` is no longer applied as a pre-trade signal filter in either
the RL training env or the backtest engine. The confidence gate in the reward
function (`_close_position()`) remains — it shapes the reward signal without
suppressing trade diversity.

Run 4 demonstrated that lower-confidence trades, when combined with proper
reward shaping, produce a healthier distribution of outcomes. Gating at 0.70+
reduced trade count without improving R:R, and removed the varied market
exposure the agent needs for the exit head to learn.

`BacktestConfig.min_confidence` field is retained for backwards compatibility
but is no longer read in `run()`.

### Change 2 — Stronger reward shaping coefficients

**Files:** `train_rl.py`, `configs/rl/sac.yaml`

```
hold_penalty_coeff:   0.002  →  0.003
early_cut_bonus_frac: 0.30   →  0.40
```

Run 4's avg hold time was still 78–80 bars for both winners and losers,
showing the exit head was not yet firing voluntarily. The 0.003 penalty
creates a stronger gradient: at 90% of max_hold the per-step cost is
`0.003 × 0.81 = 0.00243` per bar — roughly $0.24/bar at typical PnL scale,
enough to compete with the expected-value of staying in a marginal trade.

The 0.40 early-cut bonus means a voluntary close at -$200 with 20 bars
remaining on an $10/bar loss trajectory receives a bonus of
`min(20×10×0.40, 200×0.50) = min($80, $100) = $80`.

### Change 3 — Exit head curriculum warmup

**Files:** `rl_agent.py`, `train_rl.py`

```python
EXIT_THRESHOLD_WARMUP = 0.0    # first 100k steps
EXIT_THRESHOLD_FINAL  = -0.10  # after 100k steps
```

During warmup the agent exits on any `exit_logit < 0` — half the tanh output
space. This aggressively fills the replay buffer with early-exit transitions
covering a wide range of hold times and PnL levels, giving the critic the
data it needs to learn that early exits on losing trades have positive value.

After 100k steps the threshold tightens to -0.10, requiring genuine exit
conviction. The transition is logged once:

```
Curriculum: EXIT_THRESHOLD tightened 0.0 → -0.10 at step 100,000
```

Call `agent.set_step(step)` each training iteration — `train_rl.py` already
does this at the top of the training loop.

### Recommended training invocation (v3)

```bash
python scripts/train_rl.py \
  --checkpoint models/dual_branch_best.pt \
  --max-hold 80 \
  --confidence-gate 0.48 \
  --eval-every 8000 \
  --seed 42 \
  --steps 500000 \
  --mtm-scale 0.05 \
  --hold-penalty 0.003 \
  --early-cut-bonus 0.40 \
  --curriculum-warmup 100000
```

### Recommended backtest invocation (v3)

```bash
# Standard (no HITL — no confidence gate applied)
python scripts/backtest.py model=dual_branch data=xauusd

# With HITL mid-hold reviews
python scripts/backtest.py model=dual_branch data=xauusd \
  ++risk.human_exit_approval=true
```

Note: `++min_confidence=X` overrides are now ignored. Remove them from any
existing scripts or Makefile targets.

---

## Patch v4 — data expansion + obs-dim + regime research

Applied after v3 training run analysis. Three root causes addressed:
(1) 12-month single-regime dataset; (2) 90.7% hold-signal dominance;
(3) RL obs missing market-state context causing 25-trade steady-state lock-in.

### Files changed in v4

| File | Change |
|---|---|
| `scripts/download_data.py` | Full rewrite: chunked MT5, yfinance fallback, 9-symbol support |
| `src/data/feature_engineering.py` | `compute_rl_obs_features()` added |
| `scripts/train_rl.py` | obs 7→10, ATR/trend/session integrated, env updated |
| `configs/data/xauusd.yaml` | `class_weights: [2.5, 0.3, 2.5]` |
| `requirements-ta.txt` | Expanded: yfinance, MetaTrader5, full TA stack |
| `requirements-research.txt` | New: isolated notebook env |
| `notebooks/00_market_regime_explorer.ipynb` | New: 15-cell regime analysis |

### Change 1 — Dataset expansion to 6666 days (≈2008)

**File:** `scripts/download_data.py`

MT5 safety analysis: `copy_rates_range()` on a single 6666-day M1 request
silently truncates at the broker history boundary (typically 2–5yr for M1).
Solution: 30-day chunked windows with 0.5s sleep between requests and
resume support via last stored timestamp in DuckDB.

For pre-2019 history and non-price instruments (DXY, USD10Y, US500), the
script falls back to yfinance which provides unlimited D1 history back to
2000+:

```bash
# Full 6666-day history via yfinance (daily bars, all 9 tickers):
python scripts/download_data.py \
  --source yfinance \
  --regime-tickers \
  --timeframe D1 \
  --days 6666

# Recent M1 XAUUSD via MT5 (chunked, resume-safe):
python scripts/download_data.py \
  --source mt5 \
  --symbol XAUUSD \
  --timeframe M1 \
  --days 6666
```

### Change 2 — Class-weighted loss for label imbalance

**File:** `configs/data/xauusd.yaml`

The 90.7% hold-class dominance means the supervised model achieves 90%
accuracy by always predicting hold. Fix: add class weights to the
CrossEntropyLoss in supervised training:

```yaml
class_weights: [2.5, 0.3, 2.5]   # [sell, hold, buy]
```

Apply in the supervised training script:
```python
weights = torch.tensor(cfg.data.labeling.class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

The 2.5× upweight on sell/buy forces the model to learn non-hold patterns
even when they are rare. Re-tune after data expansion — the imbalance ratio
will shift with more ranging-market data (2015–2022 period).

### Change 3 — RL observation expanded 7→10 dims

**Files:** `scripts/train_rl.py`, `src/data/feature_engineering.py`

New observation vector:
```
[sell_prob, hold_prob, buy_prob,   # 0-2: supervised signal
 confidence,                        # 3: model confidence
 position_dir,                      # 4: -1/0/1
 unrealized_pnl_norm,               # 5: clipped [-5, 5]
 hold_time_norm,                    # 6: [0, 1]
 atr_norm,                          # 7: rolling ATR/close [0, 0.05] NEW
 trend_norm,                        # 8: EMA slope/close clipped [-2,2] NEW
 session_phase]                     # 9: London/NY overlap [0,1] NEW
```

`atr_norm` tells the agent current volatility regime — high ATR means
wider price swings, so holding through a dip is riskier.

`trend_norm` tells the agent whether it is trading with or against the
short-term trend — counter-trend holds should exit earlier.

`session_phase` breaks the 25-trade time-lock. The 25 trades / 2000 bars
steady-state was the agent learning pure time-based exits. With session
phase, the agent can learn that exit decisions at 0.0 (off-hours) vs 0.9
(peak London/NY) should be different.

The `ConfidenceSACAgent` `obs_dim` parameter is updated to 10. Existing
v3 checkpoints (`rl_agent_best.pt`) are **incompatible** — retrain required.

### Change 4 — Market regime explorer notebook

**File:** `notebooks/00_market_regime_explorer.ipynb`

15-cell Jupyter notebook covering:
- Data download: 9 instruments, 6666 days via yfinance
- Cross-asset overview and normalised price chart
- Rolling 252-day correlation matrix with XAUUSD
- Business cycle 4-quadrant (growth × inflation proxy from asset prices)
- Gold drivers: real yield proxy, DXY correlation, G/S ratio, G/Oil ratio
- Volatility regime detection (low / normal / high thresholds)
- HMM 3-state classification (Bear / Neutral / Bull) on XAUUSD
- K-means 4-cluster consensus cycle score across all 9 instruments
- Signal quality table: accuracy and Sharpe by regime
- Export: `data/regime/daily_regime_labels.csv` for ML pipeline

Run with:
```bash
# In research venv:
pip install -r requirements-research.txt
jupyter lab notebooks/00_market_regime_explorer.ipynb
```

### Recommended full re-training sequence (v4)

```bash
# Step 1: Expand dataset
python scripts/download_data.py --source yfinance --regime-tickers --timeframe D1 --days 6666
python scripts/download_data.py --source mt5 --symbol XAUUSD --timeframe M1 --days 6666

# Step 2: Re-train supervised model with class weights
# (add weight= to CrossEntropyLoss, see xauusd.yaml class_weights)
python scripts/train_supervised.py model=dual_branch data=xauusd

# Step 3: Re-train RL with 10-dim obs
python scripts/train_rl.py \
  --checkpoint models/dual_branch_best.pt \
  --max-hold 80 --confidence-gate 0.48 \
  --eval-every 8000 --seed 42 --steps 500000 \
  --mtm-scale 0.05 --hold-penalty 0.003 --early-cut-bonus 0.40 \
  --curriculum-warmup 100000

# Step 4: Run regime notebook
cd notebooks && jupyter lab 00_market_regime_explorer.ipynb
```
