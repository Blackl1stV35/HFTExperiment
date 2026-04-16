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
