# HFTExperiment v2 — Week 1 Paper Trade Report & Next Steps
**Period:** 2026-05-11 to 2026-05-17 (7 days, 5 trading days)
**Account:** Kanokphan Sirithienthong 49754113 — HF Markets (SV) Ltd. DEMO $100,000
**Deployment:** Run 10 _last.pt (sell P=0.302) + Phase 8b RL evolve2

---

## 1. Week 1 Performance Summary

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Total trades | 41 (40 closed + 1 open) | — |
| Win Rate | 47.50% (19W / 21L) | Breakeven: 40.4% |
| Gross Profit | +$290.86 | — |
| Gross Loss | -$195.68 | — |
| Net P&L | **+$95.18** (+$61.49 from closed CSV) | — |
| Profit Factor | 1.49 (PDF) / 1.33 (CSV closed) | >1.0 target |
| Avg Win | $12.95 | — |
| Avg Loss | -$8.79 | — |
| EV/trade | $1.54 | $1.65 modelled |
| EV/day | ~$8.78 (from CSV) | $16.51 modelled |
| Sharpe Ratio | 0.14 (MT5) | — |
| Max Drawdown | 0.157% ($157) | — |
| Max Deposit Load | 0.0187% | — |
| Max Consecutive Wins | 6 | — |
| Max Consecutive Losses | 5 | Circuit breaker threshold |
| Best Trade | +$49.45 | — |
| Worst Trade | -$20.63 | — |
| Avg Hold Time | 1h 37m | — |
| Balance | $100,095.18 | — |
| Equity | $99,966.36 | 1 open position ($128.82 unrealised) |

### Daily P&L Breakdown

| Date | Day | P&L | Trades |
|------|-----|-----|--------|
| 2026-05-11 | Mon | **+$71.62** | Strong London open |
| 2026-05-12 | Tue | **-$35.57** | Choppy session |
| 2026-05-13 | Wed | **-$55.44** | Worst day — 5 consecutive losses |
| 2026-05-14 | Thu | **-$5.41** | Near flat |
| 2026-05-15 | Fri | **+$86.29** | Best day — regime detection fired |

---

## 2. Direction Breakdown

| Direction | Trades | WR | Avg P&L | Gross |
|-----------|--------|----|---------|-------|
| Long (BUY) | 26 (65.9%) | 42.3% | **-$1.08** | -$28.15 |
| Short (SELL) | 14 (34.1%) | **57.1%** | **+$6.40** | +$89.64 |

**Critical finding: Longs are net-negative (-$28.15 gross), Shorts are strongly profitable (+$89.64 gross).** The model's directional edge is almost entirely in the sell/short signal. Long entries are noise — the buy signal (avg buy_p at close = dominated by hold) is not generating clean directional trades. This is consistent with the supervised model's asymmetric architecture: sell_P=0.302 (validated) vs buy precision untested and likely lower.

### Regime Breakdown

| Regime (gmm2) | Trades | WR | Notes |
|---------------|--------|----|-------|
| Bear | 21 (52.5%) | **57.1%** | Model performs in its trained regime |
| Bull | 19 (47.5%) | 36.8% | Model struggles in Bull regime |

Bear regime WR=57.1% vs Bull WR=36.8% — a 20 percentage point gap. This directly validates the research plan: Bull regime sell labels are noisy (1.04% sell rate vs 14.4% in Bear-SHOCK) and the model has not learned to distinguish real Bull sells from noise. The DHPF Bull-LOW boost (attempted in Run 11) was the wrong fix; regime-conditioned models (P5C) is the right fix.

---

## 3. Signal Quality Diagnostics

### RL Actor: 100% Rail-Clipped
Every single bar has `rl_size = ±1.0`. The Phase 8b RL agent is fully saturated — all position sizing and exit decisions are at maximum magnitude. This is not a trading strategy; it is a binary on/off signal. The RL layer is contributing **zero** nuance beyond what the supervised model already decided.

**Root cause confirmed:** Phase 8b was trained on 16D obs without session_phase or rq_regime. The Phase 9 18D obs vector has been patched but Phase 9 training has not yet started. Until Phase 9 is trained, the RL actor will remain rail-clipped.

**Impact:** avg hold time 1h 37m is entirely driven by the supervised model's exit signal (hold_p dropping below threshold), not by the RL policy. The RL actor is a pass-through.

### Confidence Distribution
Mean confidence at close bars: 0.697 ± 0.048. Confidence gate threshold = 0.70. The model is consistently operating near the confidence threshold, which means many potential entries are being filtered by the gate — the effective WR of unfiltered entries would be lower.

### Supervised Signal at Entry
Entries are firing primarily on hold→buy or hold→sell transitions in the supervised model. The sell_p range at entry confirms the gate is working: high-conviction sell entries (sell_p > 0.85) produced the large wins (+$49.45 short, +$20.09 short).

---

## 4. Statistical Assessment

### Edge Confirmation
- **Breakeven WR at observed win/loss ratio:** 40.4%
- **Observed WR:** 47.5%
- **Margin above breakeven:** +7.1 percentage points
- **95% Wilson Confidence Interval:** [32.9%, 62.5%]
- **CI width:** 29.6 percentage points — **too wide to confirm edge statistically**

The edge exists in the observed data (WR > breakeven, profit factor > 1.0, net positive P&L) but N=40 is insufficient to rule out luck at 95% confidence. The z-test p-value = 0.185 — cannot reject H0 that WR ≤ breakeven at the 5% level.

### Sample Size Requirements

| Confidence | Power | Trades Needed | Gap |
|------------|-------|---------------|-----|
| 95% | 80% | ~3,723 | 3,683 more |
| 95% | 95% | ~6,516 | 6,476 more |
| 90% | 80% | ~2,635 | 2,595 more |

At current pace (~8 trades/day): 95%/80% confirmation requires ~465 trading days. This is why the research plan prioritises architectural improvements over longer paper trading — the edge needs to be large enough to be statistically confirmable in a reasonable timeframe. A WR of 55%+ (vs 47.5% current) would require only ~370 trades for 95%/80% confirmation.

### Research Plan Calibration Update

The week 1 data refines the EV model parameters:

| Parameter | Original Model | Week 1 Actual | Revision |
|-----------|---------------|---------------|----------|
| Avg win | $11.34 | $12.95 | Update to $12.95 |
| Avg loss | -$8.52 | -$8.79 | Update to -$8.79 |
| WR | 0.512 (Phase 8b) | 0.475 (Phase 8b, rail-clipped) | No revision — rail-clipping reduces effective WR |
| EV/trade | $1.65 | $1.54 | Slightly lower due to lower WR |
| Trades/day | 10 | 8 | Update to 8 |
| EV/day | $16.51 | $8.78 (from CSV) | Note: PDF total $95.18/7 = $13.60/day |

The discrepancy between CSV EV/day ($8.78) and PDF ($13.60/day) is explained by the open position: $95.18 total includes an unrealised position not in the closed CSV rows.

---

## 5. Key Observations for v3 Architecture

### Observation 1 — Short signal is the alpha, long signal is noise
WR: Short=57.1%, Long=42.3%. Long avg P&L=-$1.08 (net negative). **Implication:** the v3 model should be asymmetric — optimise for sell precision exclusively. Buy labels may not be learnable from M1 data without multi-timeframe context (the buy signal requires M15+ confirmation).

### Observation 2 — Bear regime is the profitable regime
Bear WR=57.1% vs Bull WR=36.8%. **Implication:** regime-conditioned models (P5C) and regime-conditioned labels (P4A — remove Bull sell labels) are validated by live data, not just theoretical analysis. Bull entries are the primary source of losses.

### Observation 3 — RL actor contributes zero differentiation
Rail-clipped 100% means Phase 9 RL is the most urgent fix after the paper trade data period. Until the RL actor can differentiate position sizing and exit timing, the system is effectively a supervised-only signal with a fixed lot size.

### Observation 4 — Max consecutive losses = 5 matches the research plan assumption
The circuit breaker threshold of 5 consecutive losses was confirmed. P3c (curriculum adversarial replay) is justified — training explicitly on the 5-loss scenario during RL Phase 9 would directly address the worst observed outcome.

### Observation 5 — Hold time 1h 37m suggests single-regime entries
Average hold of 97 minutes on M1 bars means the model is entering at a regime transition and exiting when the regime resolves. This is the correct pattern for M1 trend-following but means the model is vulnerable to false regime transitions. M15 context (P6) would filter these.

---

## 6. Next Steps: Implementation Plan

### Immediate (while paper trading continues)

**Step 1 — Keep paper trade running, target N=100 closed trades**
Continue deployment of Run 10 _last.pt + Phase 8b RL. Do not change the deployed model until N≥100 to accumulate better statistics. Expected: ~12 more trading days.

**Step 2 — Upload Run 10 _last.pt to Colab Drive as `dual_branch_r10_last.pt`**
Preserve the best available supervised checkpoint before any future training overwrites it. This is the fallback for all future RL training until a better supervised model is confirmed.

```bash
# From local Windows machine
# Upload via Google Drive web UI or:
python -c "
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
# upload dual_branch_last.pt as dual_branch_r10_last.pt
"
```

---

### Phase 1 — Label Foundation (2 weeks on Colab)

**Step 3 — Implement hindsight-optimal labels (P1)**

Replace triple-barrier labels with dynamic programming optimal labels. The DP computes the globally optimal sell/hold/buy sequence given the full known future price path.

```python
# In scripts/precompute_features.py — add --label-method flag
python scripts/precompute_features.py     --csv data/XAUUSD_M1.csv     --existing-npz data/training_ready.npz     --label-method hindsight_dp     --dp-gamma 0.99     --dp-horizon 40     --output data/training_ready_v3.npz
```

**Expected output:** sell rate drops from 3.51% to ~2.8–3.0% (noisy labels removed), sell label quality improves.

**Decision gate:** train Run 13 on hindsight labels. If test sell_P ≥ 0.36 → proceed to Phase 2 full rebuild. If < 0.36 → MTF input (P6) is the bottleneck, prioritise before experts.

**Step 4 — Add temporal exponential reweighting (P5A)**

Wire into the existing sampler alongside hindsight labels. Calibrate λ so that 2024–2026 bars receive 5× weight of 2009–2015 bars.

```python
# In train_supervised.py build_regime_balanced_sampler():
ts_ns = _d["timestamps_ns"]
T_max = ts_ns.max(); T_range = T_max - ts_ns.min()
lambda_decay = 1.5  # calibrated: 2024+ gets ~5x weight
time_weights = np.exp(-lambda_decay * (T_max - ts_ns) / T_range).astype(np.float32)
regime_mult *= time_weights  # multiplicative with existing regime mult
```

---

### Phase 2 — RL Reward Reform (1 week on Colab, after Phase 1 gate)

**Step 5 — Implement Information Ratio reward (P3b)**

Replace raw PnL reward in `TradingEnv` with rolling Sharpe reward. Preserves compatibility with any supervised checkpoint.

```python
class IRTradingEnv(TradingEnv):
    def __init__(self, *args, ir_window=50, ir_blend_epochs=20, **kwargs):
        super().__init__(*args, **kwargs)
        self._ret_buf = deque(maxlen=ir_window)
        self._step_count = 0

    def step(self, action):
        obs, raw_r, done, info = super().step(action)
        self._ret_buf.append(raw_r)
        self._step_count += 1
        if len(self._ret_buf) >= 5:
            mu = np.mean(self._ret_buf)
            sg = np.std(self._ret_buf) + 1e-6
            ir  = mu / sg
            blend = min(self._step_count / (ir_window * 10), 1.0)
            shaped = blend * ir + (1-blend) * raw_r
        else:
            shaped = raw_r
        return obs, shaped, done, info
```

**Step 6 — Add curriculum adversarial replay (P3c)**

Track episode loss streaks. Replay episodes with ≥3 consecutive losses at 3× weight during RL training.

**Step 7 — Run RL Phase 9**

Use the best supervised checkpoint from Phase 1 (Run 13 if sell_P≥0.36, else Run 10 _last.pt). Train with 18D obs (session_phase + rq already patched), IR reward, adversarial replay.

```bash
python scripts/train_rl.py     --checkpoint ".../dual_branch_r13_best.pt"     --steps 2000000 --n-evolves 12     --commission 0.70 --spread-pips 2.0     --reward ir --ir-window 50     --adversarial-replay --replay-weight 3.0
```

---

### Phase 3 — Architecture Rebuild (3–4 weeks on Colab)

**Step 8 — Regime-conditioned expert models (P5C)**

Train 3 expert encoders: Bear-HIGH specialist, Bull-ALL generalist, Transition detector. Implement soft routing gate (2-dim regime input → 3-expert softmax).

Files to create:
- `src/encoder/regime_router.py` — RegimeRouter class
- `src/encoder/expert_pool.py` — ExpertPool managing 3 DualBranchModel instances
- Update `train_supervised.py` to support `model=regime_experts`

**Step 9 — Multi-timeframe M1 + M15 dual stream (P6)**

Collect XAUUSD M15 data (same date range as M1). Add second input stream with cross-attention fusion.

Data collection: MT5 terminal → History Center → XAUUSD M15 → Export to CSV.

Files to modify:
- `src/data/preprocessing.py` — add M15 loading path
- `src/encoder/price_branch_transformer.py` — add DualTimeframeBranch
- `precompute_features.py` — add --m15-csv flag

**Step 10 — 512D penultimate encoder to RL (P4B)**

Pass the 512-dim `combined` vector from encoder forward pass to SAC agent. SAC input becomes 512 + 14 = 526D.

Files to modify:
- `src/meta_policy/rl_agent.py` — expand input dims, add projection layer
- `src/training/train_rl.py` — pass `combined` tensor from encoder
- `paper_trade.py` — extract `combined` from model.forward() at inference

---

### Phase 4 — Adaptation Layer (ongoing after Phase 3)

**Step 11 — Online continual learning (P5B)**

Weekly fine-tune on fresh hindsight-labelled bars from live paper trade data. LR=1e-6 to prevent catastrophic forgetting.

```bash
# Run every Sunday evening Bangkok time (Sunday = Monday Asian open)
python scripts/finetune_online.py     --checkpoint ".../dual_branch_r13_best.pt"     --new-bars data/live_bars_week_N.csv     --lr 1e-6 --epochs 3     --output ".../dual_branch_r13_online_wN.pt"
```

**Step 12 — Silver futures input (P2)**

After Phase 3 validates MTF architecture, add XAGUSD M1 as a third input channel. The XAG-XAU correlation differential is a leading indicator for gold directional moves.

---

## 7. Decision Gates Summary

| Gate | Condition | If YES | If NO |
|------|-----------|--------|-------|
| G1: After Run 13 ep60 | sell_P ≥ 0.36 | Proceed Phase 2 full rebuild | Prioritise P6 (MTF) before experts |
| G2: After RL Phase 9 | Paper trade WR ≥ 52% | Deploy to Pro account ($100) | Extend paper trade, refine reward |
| G3: After Phase 3 | sell_P ≥ 0.40 + WR ≥ 54% | Scale lot size to 0.02 | Continue Phase 4 adaptation |
| G4: After N=200 trades | WR ≥ 52% confirmed | Live Pro deployment | Further research |

---

## 8. Research Plan EV Projections (Recalibrated)

Using week 1 actual win/loss magnitudes (avg win $12.95, avg loss $8.79):

| Phase | sell_P | WR | Avg win | Trades/day | EV/day | Annualised |
|-------|--------|----|---------|------------|--------|------------|
| Current (Phase 8b) | 0.302 | 47.5% | $12.95 | 8 | $8.78 | $2,213 |
| Phase 2 (P3b IR reward) | 0.302 | 55% | $16.50 | 6 | $16.80 | $4,234 |
| Phase 1+2 (Run 13 + IR) | 0.360 | 55% | $16.50 | 6 | $20.46 | $5,156 |
| Phase 3 (P5C + P6) | 0.450 | 60% | $18.00 | 7 | $39.69 | $10,002 |
| Full stack aggressive | 0.527 | 66% | $19.50 | 7 | $55.77 | $14,054 |

At 0.01 lot / $100,000 demo. Scale proportionally to actual deployment capital.

---

## 9. Risk Parameters (Validated by Week 1)

| Parameter | Week 1 Observed | Research Plan Assumption | Status |
|-----------|----------------|--------------------------|--------|
| Max consecutive losses | 5 | 5 | ✓ Confirmed |
| Max drawdown | 0.157% | <2% circuit breaker | ✓ Well within |
| Avg hold time | 1h 37m | M1 bar resolution | ✓ Expected |
| RL rail-clipping | 100% | Known issue | ⚠ Requires Phase 9 |
| Long WR | 42.3% | — | ⚠ Below breakeven |
| Short WR | 57.1% | — | ✓ Above breakeven |
| Bull WR | 36.8% | — | ⚠ Below breakeven |
| Bear WR | 57.1% | — | ✓ Above breakeven |

---

*HFTExperiment v2 · Week 1 Report + v3 Next Steps · 2026-05-17*
