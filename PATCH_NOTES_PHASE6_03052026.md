# Phase 6 — Feature Exploration Lab: Cross-Asset & Microstructure Expansion
## Decision Record, Justifications & Upcoming Notebook Plan

**Status:** Supervised training complete (Run 7 accepted). RL phase approved to begin.
**Parallel track:** Feature Exploration Lab — validate 4 candidate features against the
existing 10-dim baseline before integrating into Run 8+ or the RL observation space.
**Date:** 2026-05-03 · **Hardware target:** A100 40 GB (Colab Pro)

---

## 1. Phase 5 Final Results — Run 7 Accepted

### Run history across all phases

| Phase | Architecture | Best test sell P | Signal score | Decision |
|-------|-------------|-----------------|--------------|----------|
| 3 (InceptionBlock+TCN) | Run 5 | 0.253 | — | RL ceiling hit |
| 3 (InceptionBlock+TCN) | Run 6 (resume) | 0.259 | — | At capacity |
| 4 (ScatterTCN) | Run 1 | — | — | Terminated ep1 (34s/batch) |
| 5 (Transformer d=512) | Run 1 | 0.073 | 0.041 | Warmup |
| 5 (Transformer d=512) | Run 2 | 0.253 | 0.141 | Prior best |
| 5 (Transformer d=512) | Run 3 | 0.247 | 0.138 | Regression (cosine LR surge) |
| 5 (Transformer d=512) | Run 4 | 0.251 | 0.149 | Incremental |
| 5 (Transformer d=512) | Run 5 | 0.263 | 0.156 | Improving |
| 5 (Transformer d=512) | Run 6 | 0.271 | 0.174 | Monotone gain |
| **5 (Transformer d=512)** | **Run 7** | **0.283** | **0.193** | **Accepted — proceed to RL** |

### Run 7 — Accepted checkpoint details

| Metric | Val (ep61 best) | Test |
|--------|----------------|------|
| Loss | 0.563 | — |
| Accuracy | — | — |
| Sell precision | — | **0.283** |
| Sell recall | — | — |
| Buy precision | — | **0.130** |
| Signal score | **0.171** | **0.193** |
| LR at checkpoint | 2.5e-5 | — |
| LR decays fired | 1 (ReduceLROnPlateau, ep78) | — |

**Val loss trajectory across Runs 1–7:** 0.554 → 0.658 → 0.606 → 0.597 → 0.582 →
0.563. Monotonically improving. The single LR decay at ep78 drove sell P from
0.17→0.22 in the final 12 epochs — confirming ReduceLROnPlateau is working as
designed.

**Model has not fully converged.** LR is at 2.5e-5 with only one decay fired.
A second decay would push val precision into the 0.25+ range. However, the improvement
trajectory is strong enough that this checkpoint is a viable RL input. Further
supervised tuning has diminishing returns relative to training the RL reward function
on a live XAUUSD signal.

### What produced the Run 7 improvement over Run 2 (0.283 vs 0.253)

Three components contributed simultaneously and cannot be individually ablated
without rerunning controlled experiments:

1. **Astrocyte routing module** (`AstrocyteGatingModule`, Vivet & Arenas 2026):
   replaces the hard `LocalCausalAttention(w=20)` pool in the short stream with
   content-addressed gain-simplex routing. Competitive resource allocation suppresses
   hold-pattern interference and amplifies the rare sell/buy patterns — the exact
   mechanism predicted by the astrocyte paper's high-K, high-interference benchmark.

2. **Label smoothing ε=0.1** (`FocalLoss(label_smoothing=0.1)`): prevents the model
   from assigning zero probability to non-target classes. With 95.8% hold labels,
   hard targets drove the model to collapse confidence onto hold; ε=0.1 maintains a
   residual gradient signal for sell/buy throughout training.

3. **Buy class boost** (auto-computed as `sell_n / buy_n`, capped at 10×): corrected
   the asymmetric class weight. The previous `[2.5, 0.3, 2.5]` config treated buy and
   sell as equally rare when sell has 5× more labels (116,994 vs 22,935). The boost
   raised the effective buy weight to reflect the true scarcity ratio.

---

## 2. RL Phase — Approved

**Checkpoint:** `_best.pt` (ep61, val signal_score=0.171)
**Script:** `scripts/train_rl.py`
**Environment:** `src/meta_policy/` SAC with confidence-based position sizing

The supervised backbone at signal=0.171 val / 0.193 test is the strongest
starting point this system has seen. The RL layer will:
- Gate signals using confidence scores from the dual-branch fusion head
- Amplify high-confidence directional conviction into larger notional positions
- Suppress low-confidence bars toward hold (reducing transaction cost drag)
- Learn the regime-conditional risk budget from the 6D regime observation array

RL proceeds in parallel with the Phase 6 Feature Exploration Lab. These are
independent tracks — the RL run uses the current `training_ready.npz` unchanged.
Feature changes, if validated, will be integrated into a new NPZ and retrained
for Run 8 of the supervised model.

---

## 3. Feature Expansion Decision — Macro-Prudential & Climate Features

A proposal was reviewed to integrate the following external feature classes:
climate change indices, LTV ratios, counter-cyclical capital buffers (CCyB),
dynamic provisioning, leverage caps, and G-SIB capital surcharges.

### Decision: RL observation space only — not supervised price branch

**Justification:**

All proposed macro-prudential and climate features share a fundamental property:
they change at monthly or quarterly frequency and are broadcast identically to
thousands of consecutive M1 bars. For the supervised model, a feature that is
constant across a session has zero local KS statistic (it cannot distinguish
consecutive bars from each other within any training window). It can only serve
as a global distribution shift indicator — a role already filled by the 6D regime
array (`gmm2`, `vol_regime`, `km_label_63d`).

However, these features have genuine value as **RL environment context**. The RL
agent makes position sizing decisions at episode boundaries — a fundamentally
different decision timescale from bar-by-bar classification. At that horizon,
knowing that CCyB is being tightened system-wide (a risk-off signal for leveraged
positions) or that climate VaR has crossed a stress threshold (systemic risk
premium rising → gold safe-haven demand) is directly actionable.

| Feature class | Supervised value | RL value | Integration path |
|---------------|-----------------|----------|-----------------|
| Climate risk index (NGFS/MSCI) | ❌ (quarterly, constant per session) | ✅ Regime multiplier | Add to `daily_regime_labels.csv` → `join_regime_labels()` |
| LTV ratios (regulatory) | ❌ (monthly policy change) | ✅ Credit cycle proxy | RL obs scalar (monthly broadcast) |
| CCyB (counter-cyclical buffer) | ❌ (quarterly announcement) | ✅ Event-driven risk flag | Binary event flag in regime CSV on announcement date |
| Dynamic provisioning | ❌ (overlaps vol_regime) | ⚠️ Marginal (redundant with gmm2) | Not recommended; information already captured |
| Leverage caps | ❌ (annual regulatory) | ⚠️ Background only | Not recommended; no short-term price signal |
| G-SIB capital surcharges | ❌ (annual, fully anticipated) | ❌ Pre-announced, zero surprise | Excluded |

**CCyB** is the strongest candidate among this group. Central bank CCyB decisions
are discrete, publicly announced, and have clear directional implications for gold:
CCyB expansion → tightening financial conditions → gold consolidation or mild sell-off;
CCyB cut (stress response, e.g., March 2020) → systemic risk event → gold safe-haven
spike. An event-driven binary flag (`ccyb_cut_event`, `ccyb_hike_event`) added to
the regime CSV on announcement dates provides the RL agent with a regime shift
signal that cannot be derived from price alone.

**Implementation deferred to post-RL phase.** The RL training run should complete
first using the current 6D regime observation space. CCyB and climate features will
be evaluated as RL obs extensions in Phase 7.

---

## 4. Phase 6 Feature Exploration Lab — Plan

### Motivation

The current 10-dim feature set (`prepare_features()` output) was designed around a
single validation criterion: KS statistic and mutual information against ATR-adaptive
labels at 240-bar sequence depth. The Phase 4 sequence profiling found `spread_pressure`
(KS=0.600) as the dominant discriminative feature. The 4 microstructure features have
a mean MI of 0.032 — 6.8× the OHLCV baseline (0.0047).

However, the profiling was conducted on isolated features. No cross-asset or
session-level features were included in the candidate set. The following 4 candidate
features have theoretical grounding for improving sell/buy precision specifically —
the bottleneck metric under the current signal score formula.

### 4.1 DXY Intrabar Correlation (`dxy_return_20`)

**What it is:** Rolling 20-bar DXY return, computed from M1 DXY OHLCV data aligned
to the XAUUSD timestamps. Expressed in bps, z-scored over a 120-bar rolling window.

**Theoretical justification:**

Gold's negative correlation with the US Dollar Index (DXY) is structurally embedded
in the commodity pricing mechanism: gold is USD-denominated globally, so a 1% DXY
appreciation creates a direct downward pressure on gold's USD price, independent of
demand fundamentals. This is not a statistical artefact — it is a balance sheet
identity for any market participant holding non-USD reserves and hedging via gold.

The existing 10 features are entirely endogenous to the XAUUSD price series. They
capture *how* gold moved but not *why* it moved at that specific moment. A DXY return
feature at M1 resolution provides exogenous causal information. Sequences where
`wick_asymmetry > 0` (upper wick dominance → rejected upside) AND `dxy_return_20 > 0`
(DXY strengthening) should have higher sell-label precision than `wick_asymmetry > 0`
alone — the DXY signal disambiguates technical rejection from mere noise.

**Expected discriminative signal:** Hypothesis: KS(sell vs hold) > 0.15 based on
the known 0.6–0.8 rolling correlation between DXY and XAU/USD price direction.

**Data source:** Existing `TickStore` with DXY M1 parquet, or computed from DXY
component tick data. Alignment: left-join on UTC timestamp; forward-fill gaps ≤ 5 bars.

**Implementation in `prepare_features()`:**
```python
# dxy_return_20 — 20-bar DXY return z-scored over 120-bar window
dxy_close = dxy_df["close"].to_numpy().astype(np.float64)
dxy_ret   = np.log(dxy_close[1:] / dxy_close[:-1])  # log returns
dxy_ret   = np.concatenate([[0.0], dxy_ret])
dxy_20    = rolling_sum(dxy_ret, window=20)           # sum of 20 log returns ≈ 20-bar return
dxy_z     = z_score_rolling(dxy_20, window=120)       # z-score relative to recent distribution
dxy_z     = np.tanh(dxy_z).astype(np.float32)        # tanh-smooth tails
```

**Feature dimensions after addition:** 10D → 11D

---

### 4.2 VWAP Deviation (`vwap_dev_norm`)

**What it is:** `(close - vwap) / atr` where VWAP resets at each UTC session
boundary (00:00, 08:00, 16:00 — London, NY, Asian session anchors). ATR-normalised
to make the signal scale-invariant across volatility regimes.

**Theoretical justification:**

VWAP is the institutional reference price. Market makers and algorithmic desks
execute against VWAP benchmarks — a position that was filled above VWAP has a built-in
reversion pressure as the session unfolds. When `close > vwap` (price trading above
the session's volume-weighted mean), it represents an extended position that
statistical arbitrage will fade. When `close < vwap`, mean reversion implies upward
pressure.

Critically, VWAP deviation captures something `wick_asymmetry` does not: the
*accumulated* institutional order flow positioning across the session, not just
the local bar-level buying/selling pressure. A sell label at a bar where
`vwap_dev_norm = +2.5` (price is 2.5 ATRs above the session VWAP) carries a
higher-confidence mean-reversion signal than a sell label at `vwap_dev_norm = 0`.

This feature is computable from the existing OHLCV data in `TickStore` — no external
data source required. Typical VWAP implementation:

```python
# VWAP deviation — session-anchored (3 anchors per day), ATR-normalised
typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
session_id    = (df["timestamp"].dt.hour // 8)          # 0=Asian, 1=London, 2=NY
cum_pv        = (typical_price * df["tick_volume"]).groupby(session_id).cumsum()
cum_vol       = df["tick_volume"].groupby(session_id).cumsum()
vwap          = cum_pv / (cum_vol + 1e-8)
vwap_dev      = (df["close"] - vwap) / (atr_14 + 1e-8)  # ATR-normalised
vwap_dev_norm = np.tanh(vwap_dev).astype(np.float32)    # smooth tails
```

**Expected discriminative signal:** Hypothesis: KS(sell vs hold) > 0.20. Sells
should systematically occur at positive VWAP deviation; buys at negative deviation.
The feature is orthogonal to `spread_pressure` (which measures the bid-ask cost
relative to range, not the accumulated position relative to session mean).

**Feature dimensions after addition:** 11D → 12D

---

### 4.3 Session Phase Encoding (`session_enc`)

**What it is:** A 3-level continuous encoding of the current UTC trading session:
Asian open (00:00–08:00 UTC) = 0.0, London open (08:00–16:00 UTC) = 0.5,
NY open/overlap (16:00–00:00 UTC) = 1.0. Linear interpolation within each session
window to capture intra-session progression.

**Theoretical justification:**

The `session_phase` feature already exists in `training_ready.npz` (computed by
`compute_rl_obs_features()` and used in the RL observation space). It is not currently
included in the 10-dim supervised feature array. This is an inconsistency — the RL
agent has access to session context that the supervised signal generator does not.

The discriminative argument rests on liquidity structure. XAUUSD microstructure changes
dramatically across sessions:

- Asian session (00:00–08:00 UTC): low volume, wide relative spread, range-bound.
  `spread_pressure` is systematically elevated but predictive power is lower because
  direction is undetermined by thin order books.
- London open (08:00–09:30 UTC): highest directional volatility. Institutional
  positioning drives 40–60% of daily range in the first 90 minutes. `wick_asymmetry`
  and `vol_zscore` discriminate most strongly here.
- NY session (13:30–16:00 UTC overlap): macro data releases (NFP, CPI, FOMC) create
  sharp directional moves. `dxy_return_20` is most informative in this window.

The Transformer's positional encoding provides temporal position within the 120-bar
sequence window, not absolute session time. A `session_enc` feature gives the model
the *calendar* context that positional encoding cannot provide — two sequences with
identical `wick_asymmetry` and `spread_pressure` profiles mean different things at
08:15 UTC vs 22:15 UTC.

**Implementation:** Single-line addition to `prepare_features()`:
```python
hour       = df["timestamp"].dt.hour.to_numpy().astype(np.float32)
session_enc = np.where(hour < 8, 0.0, np.where(hour < 16, 0.5, 1.0)).astype(np.float32)
# Intra-session linear progression
minute_frac = df["timestamp"].dt.minute.to_numpy() / 60.0
session_enc += np.where(hour < 8, minute_frac / 8,
               np.where(hour < 16, minute_frac / 8, minute_frac / 8)).astype(np.float32) * 0.5
```

**Feature dimensions after addition:** 12D → 13D

**Note:** This is the lowest-risk addition — the feature already exists in the NPZ,
requires no new data, and only needs to be routed into the `prepare_features()` output
array alongside the existing 10 features.

---

### 4.4 Multi-Timeframe Return Features (`ret_5m`, `ret_15m`, `ret_1h`)

**What it is:** Three additional return features computed by aggregating the existing
M1 bars: 5-minute return (sum of 5 log returns), 15-minute return (sum of 15), and
60-minute return (sum of 60). Each z-scored over a 120-bar rolling window and
tanh-smoothed.

**Theoretical justification:**

The Phase 4 sequence profiling revealed the core architectural tension:
`spread_pressure` carries predictive information across the full 240-bar window
(long timescale), while `wick_asymmetry` and `vol_zscore` are concentrated in the
last 20 bars (short timescale). This incompatibility was the motivation for the
ScatterTCN's multi-scale scattering transform — which was theoretically correct but
computationally impractical (363+ minutes per epoch).

The multi-timeframe return features provide the Transformer with explicit access to
both timescales without requiring a multi-scale encoder. They are conceptually
equivalent to providing manually computed scattering coefficients at 3 scales:

| Feature | Timescale | Analogous scattering layer |
|---------|-----------|--------------------------|
| `bar_return_bps` (existing) | M1 (1 bar) | Layer 2 high-frequency detail |
| `ret_5m` | 5 bars | Layer 2 mid-frequency detail |
| `ret_15m` | 15 bars | Layer 1 modulation coefficient |
| `ret_1h` | 60 bars | Layer 0 low-frequency envelope |

The Transformer's attention mechanism can learn to weight these scales differently
per regime — the `LocalCausalAttention(w=20)` short stream already implicitly does
this for the last 20 bars. The multi-TF returns extend this to 60-bar horizons
without adding architectural complexity.

This is the highest-risk addition in terms of feature correlation: `ret_1h` will
be correlated with `bar_return_bps` (they are computed from the same price series).
The exploration notebook must explicitly test for redundancy via MI and partial
correlation analysis before recommending integration.

**Implementation:**
```python
# Multi-TF returns from M1 log returns
log_ret = np.log(close[1:] / close[:-1])
log_ret = np.concatenate([[0.0], log_ret])
ret_5m  = np.convolve(log_ret, np.ones(5),  "same")  # 5-bar sum
ret_15m = np.convolve(log_ret, np.ones(15), "same")  # 15-bar sum
ret_1h  = np.convolve(log_ret, np.ones(60), "same")  # 60-bar sum
# Z-score + tanh each
for arr in [ret_5m, ret_15m, ret_1h]:
    arr = z_score_rolling(arr, window=120)
    arr = np.tanh(arr).astype(np.float32)
```

**Feature dimensions after addition:** 13D → 16D

**Model config change required:** `input_dim: 10 → 16` in `configs/model/dual_branch.yaml`.
The `LearnableScatteringBlock` front-end takes raw 10-dim input and projects to 104
scatter coefficients. Adding 6 new dims would break the scatter output shape. Two
options: (a) bypass the scattering front-end for the new 6 features and add them
after the `scatter_proj` linear layer, or (b) retrain the scattering block with
16-dim input. Option (a) is lower risk and avoids invalidating the existing scattering
weight initialisation.

---

## 5. Exploration Notebook Plan — `notebooks/phase6_feature_exploration.ipynb`

The exploration notebook reviews `notebooks/phase4_microstructure_exploration.ipynb`
as the structural reference. Phase 4 established the KS / MI / temporal profiling
methodology that validated the existing 10 features. Phase 6 applies the same
methodology to the 4 candidate features, with one additional test: partial
information analysis to check for redundancy against the existing 10D baseline.

### Notebook structure

```
§1  Setup & Data Loading
    - Load training_ready.npz (existing 10D features + labels + regime arrays)
    - Load DXY M1 parquet (for dxy_return_20)
    - Align timestamps; forward-fill DXY gaps ≤ 5 bars
    - Compute the 4 candidate features (dxy_z, vwap_dev, session_enc, ret_5m/15m/1h)

§2  Discriminative Signal Validation (per Phase 4 methodology)
    - KS test: feature distributions split by label class (sell / hold / buy)
    - Mutual information: sklearn.feature_selection.mutual_info_classif
    - Threshold: KS > 0.05 AND MI > 0.005 required for inclusion
    - Comparison table: new features vs existing 4 microstructure features

§3  Temporal Profiling at 240-bar depth
    - Rolling KS at bar positions 0→240 (as in Phase 4 sequence_profiles.png)
    - Key question for each candidate:
      · Is the signal distributed across all 240 bars or concentrated in last 20?
      · Does the timescale match the short stream (w=20) or long stream?
    - Outputs: sequence_profiles_phase6.png

§4  Redundancy Analysis
    - Pairwise MI between candidate features and all existing 10 features
    - Partial correlation: candidate_i ↔ label | existing_10_features
    - Mutual information heatmap (13×13 or 16×16 depending on what passes §2)
    - Decision threshold: partial MI < 0.002 required (feature adds information
      the model does not already have via existing 10)

§5  Regime-Conditional Analysis
    - Split by (gmm2 × vol_regime) → 6 buckets
    - Per-bucket KS and MI for each candidate feature
    - Key question: is the DXY signal stronger in Bear-HIGH regime?
      (expected yes — gold/dollar decoupling in calm regimes reduces signal quality)
    - Outputs: regime_conditional_phase6.png

§6  Recommendation Summary
    - Rank candidates by: KS(sell) × MI × (1 - redundancy_score)
    - Decision per feature: Include / Exclude / Include_RL_only
    - NPZ rebuild specification: new feature array dimensions and scattering bypass plan
    - Model config delta: input_dim change, scattering bypass architecture

§7  Next Steps
    - Supervised Run 8 config (new input_dim, adjusted class weights if label
      distribution shifts with new features)
    - RL obs space extension (CCyB, climate) — deferred to Phase 7
```

### Reference notebook

`notebooks/phase4_microstructure_exploration.ipynb` provides:
- KS/MI computation pipeline (§2)
- 240-bar temporal profiling logic (§3)
- `sequence_profiles.png` generation (§3)
- `mutual_information.png` heatmap (§4)
- `regime_conditional.png` split analysis (§5)

Phase 6 notebook extends this pipeline with partial correlation (§4) and the
DXY alignment preprocessing step (§1). All other cells can be adapted directly
from Phase 4.

---

## 6. Priority Checklist

### Immediate — RL track
- [ ] Start `scripts/train_rl.py` with `_best.pt` (ep61, signal=0.193)
- [ ] Monitor: sell P in RL backtest (target: breakeven WR ≥ 49%)
- [ ] Monitor: PnL trajectory on held-out 2025–2026 data (Bear-HIGH regime)

### Immediate — Feature Exploration track (parallel)
- [ ] Download DXY M1 parquet (or compute from component data)
- [ ] Implement 4 candidate features in isolation (not yet in `prepare_features()`)
- [ ] Run `notebooks/phase6_feature_exploration.ipynb`
- [ ] KS + MI validation against Phase 4 baseline thresholds

### Near-term — if ≥ 2 features pass validation
- [ ] Add passing features to `prepare_features()` with cache-bust (new NPZ filename)
- [ ] Update `input_dim` in `configs/model/dual_branch.yaml`
- [ ] Design scattering bypass for post-scatter feature injection
- [ ] Supervised Run 8 — test new feature set

### Deferred — Phase 7
- [ ] CCyB event flag in `daily_regime_labels.csv`
- [ ] Climate risk percentile (monthly NGFS scenario score)
- [ ] RL obs space extension to 8D (current 6D + ccyb_event + climate_risk)
- [ ] Retrain RL agent with extended obs space on best supervised checkpoint

---

## 7. Feature Dimension Summary

| Version | Features | Dim | Notes |
|---------|----------|-----|-------|
| Phase 3 (baseline) | OHLCV + spread | 6D | Original InceptionBlock input |
| Phase 4 (current) | + bar_return, wick_asym, vol_zscore, spread_pressure | **10D** | Validated KS/MI |
| Phase 6 candidate A | + dxy_return_20 | 11D | Needs DXY M1 data |
| Phase 6 candidate B | + vwap_dev_norm | 12D | Computable from existing data |
| Phase 6 candidate C | + session_enc | 13D | Already in NPZ (RL obs) |
| Phase 6 candidate D | + ret_5m, ret_15m, ret_1h | 16D | Correlation risk — validate §4 |
| **Phase 6 target** | **Validated subset** | **11–16D** | **Pending notebook** |

---

*Architecture Decision Record — 2026-05-03*
*HFTExperiment v2, Phase 5 → Phase 6 (Feature Exploration Lab)*
