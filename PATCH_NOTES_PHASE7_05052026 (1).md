# Phase 7 — Feature Signal Quality Lab: Patch Notes & Pipeline Change Proposal
## Exploration Results, Architecture Decision Record & `precompute_features.py` Change Request

**Status:** Label-free exploration complete (Phase 7 v1 + v2). Phase 8 candidate set confirmed.
**Date:** 2026-05-05 · **Authored by:** Feature Exploration Lab
**Addressed to:** Master agent / pipeline maintainer

---

## 1. Phase 7 Scope & Objective

Phase 7 was a label-free signal quality lab — a diagnostic layer between the Phase 6
feature validation (which used labels and KS/MI/redundancy criteria) and Phase 8 (which
will run the full validation pipeline on a refined candidate set). The objective was to
characterise the intrinsic signal quality of 5 theory-grounded candidates before
committing to label-based validation.

**Candidates derived from:**
- Hagströmer, Henricsson & Nordén (2016) — *Components of the bid–ask spread and variance*
- El Euch, Fukasawa & Rosenbaum (2016) — *Microstructural foundations of leverage effect
  and rough volatility*

**Notebook versions run:** `phase7_signal_quality_lab.ipynb` (v1) and
`phase7v2_signal_quality_lab.ipynb` (v2 with methodology corrections)

---

## 2. Candidate Feature Outcomes

### 2.1 Summary verdict

| Feature | Theory basis | v1 status | v2 status | Phase 8 |
|---------|-------------|-----------|-----------|---------|
| `adverse_selection_proxy` | Hagströmer §2.3 | Ready | Confirmed | ✅ Include |
| `inventory_pressure` | Hagströmer §2.4 | kurt=224.9 | kurt=302→148 (all versions) | ❌ Blocked — data issue |
| `order_processing_residual` | Hagströmer §2.5 | Ready | max\|r\|=0.087 | ✅ Include |
| `hawkes_excitation_5` | El Euch τ=empirical | τ=5 (wrong) | τ=96 confirmed | ✅ Include |
| `roughness_rv` / `roughness_indicator` | El Euch + fractal | H=0.51 (wrong method) | H=0.83 (volatility clustering) | ❌ Exclude |

### 2.2 `adverse_selection_proxy` — confirmed clean

Direction balance validated: 51.4% buy / 48.6% sell across 5,680,771 bars, balanced
in both Bear (mean direction +0.028) and Bull (+0.027) regimes. The mean=−0.072 in the
feature (post z-score) reflects genuine downside momentum in gold's permanent price
impact — a real empirical property, not a sign error.

ACF decays within ~10 bars. Max external correlation 0.278 vs `bar_return_bps` (expected
— both log-return based; the direction multiplication makes them non-redundant). Ready
for Phase 8.

### 2.3 `order_processing_residual` — most orthogonal feature produced

Max external |r| = 0.087 — the most orthogonal candidate across Phase 6 and Phase 7
combined. The Hagströmer decomposition (spread − 0.5×adverse − 0.25×inventory) is
extracting information not available in any of the existing 10D features. ACF decays
within ~15 bars. Ready for Phase 8.

### 2.4 `hawkes_excitation_5` — τ confirmed distinct from `ret_5m`

v1 found empirical τ=5 using a random-sampled (non-consecutive) ACF — this was a
methodology error (gaps in sampled bars corrupt lag-1 structure). v2 with gap-free
consecutive-segment ACF found empirical τ=96 bars (~1.6 hours). All τ variants are
distinct from `ret_5m` (max r=0.716 at τ=3, 0.412 at τ=5, falling monotonically).
`hawkes_excitation_5` is clean (skew=0.01, kurt=−1.26), ready for Phase 8. A τ=96
variant should be added as a comparison candidate in Phase 8 given the corrected
empirical kernel estimate.

### 2.5 `roughness_rv` — excluded, wrong quantity

Two-iteration diagnosis confirmed the root cause. v1 measured H via R/S on raw log_ret
→ H≈0.51 (Brownian, correct for efficient market prices). v2 measured H via R/S on
log(Realised Variance 60-bar) → H≈0.83. This is not rough volatility — it is
**volatility clustering** (high RV follows high RV), the GARCH/stochastic vol effect.
El Euch et al.'s H≈0.10 applies to the volatility process estimated from daily
log-realised-variance series over multi-year windows at tick frequency. At M1 resolution
this quantity is not computable from a 60-bar RV window. DFA on log_ret also failed
(negative H values — implementation incompatibility with the tanh-normalised series).
Excluded from Phase 8. No further revision recommended without a fundamental change in
data frequency or methodology.

---

## 3. `inventory_pressure` — Root Cause Analysis & Pipeline Change Proposal

### 3.1 Failure history

Three implementations were attempted across v1 and v2, all failing the kurtosis target
(|kurt| < 5):

| Version | Input | Method | Kurtosis | Root cause |
|---------|-------|--------|----------|------------|
| v1 | `tickvol_sc` (col 4) | Session cumsum / session vol | 224.9 | MinMax-scaled, absolute magnitude lost |
| v2 | `spread_sc` (col 5) | Session cumsum / session |spread| | 144.3 | Same MinMax magnitude problem |
| v3 | `vol_zscore` (col 8) | Session cumsum / bar count | 302.5 | Zero-inflation: vol_zscore ≈ 0 for most M1 bars |
| v4 | `vol_zscore` (col 8) | Rolling 20-bar mean | 148.2 | Same zero-inflation, diluted differently |

### 3.2 Zero-inflation diagnosis

`vol_zscore` (col 8) is computed in `preprocessing.py` as:

```python
vol_zscore = np.tanh(((vol - rolling_mean_20) / rolling_std_20) / 2)
```

For XAUUSD M1, tick volume is near-constant during the Asian session (00:00–08:00 UTC)
and at weekend boundaries. The 20-bar rolling z-score of a near-constant series is
near-zero. The `/2` divisor inside tanh further compresses the distribution. Result:
`vol_zscore` is 0.000 for the majority of bars, with rare large spikes at London open
and macro release events. Rolling any function of this signal produces a zero-inflated
distribution where kurtosis explodes as σ → 0 while the 4th moment is driven by spikes.

This is not fixable by changing the accumulation method. The zero-inflation is at the
`vol_zscore` computation level, which correctly reflects the actual tick volume
distribution. No rescaling, window change, or normalization approach resolves it.

### 3.3 Why all 10D array columns fail

| Column | Content | Problem for inventory proxy |
|--------|---------|----------------------------|
| col 4 `tickvol_sc` | 120-bar rolling MinMax of tick_volume | Absolute magnitude lost; all values ∈ [0,1] within window |
| col 5 `spread_sc` | 120-bar rolling MinMax of spread | Same magnitude loss; spread ≠ volume proxy |
| col 6 `bar_return_bps` | (close[t]−close[t-1])/close[t-1]×10000 | Collapses to `adverse_selection_proxy` when × direction |
| col 7 `wick_asymmetry` | (upper−lower wick)/range | Geometric feature; not volume |
| col 8 `vol_zscore` | tanh(z-score/2) of tick_volume over 20 bars | Zero-inflated; std≈0.03 |
| col 9 `spread_pressure` | log1p(spread/range) | Pure cost metric; no directional component |

**No column in the existing 10D feature array carries the absolute tick_volume
magnitude required to compute a well-behaved inventory pressure proxy.**

### 3.4 Correct implementation (requires NPZ change)

The Volume Imbalance Order (VIO) formula from microstructure literature:

```
VIO[t] = Σ_{k=t-W+1}^{t} (V_k × sign_k) / Σ_{k=t-W+1}^{t} V_k
```

where `V_k` = raw tick volume and `sign_k` = trade direction (±1). This is bounded to
`(-1, 1)` by construction, has no zero-inflation (denominator = total volume, always
positive), and correctly captures the session-level order flow imbalance that
Hagströmer et al. identify as the inventory component.

```python
# Correct VIO — requires tick_volume_raw from NPZ
tick_vol_raw = _d['tick_volume_raw'].astype(np.float64)   # NEW NPZ key
signed_vol   = tick_vol_raw * trade_direction
kernel_W     = np.ones(W) / W                             # W = 20 bars
cum_signed   = np.convolve(signed_vol,   kernel_W, mode='full')[:N]
cum_total    = np.convolve(tick_vol_raw, kernel_W, mode='full')[:N]
inventory_pressure = np.tanh(cum_signed / (cum_total + 1e-8)).astype(np.float32)
# Distribution: well-behaved, std ≈ 0.15–0.25, |skew| < 1.0, kurt < 5 (predicted)
```

---

## 4. Pipeline Change Request — `precompute_features.py`

**Priority:** Medium (deferred to post-Phase 8 NPZ rebuild)
**Blocking:** `inventory_pressure` feature only — Phase 8 proceeds without it

### 4.1 Change description

Add `tick_volume_raw` (unscaled tick volume) to the `savez_compressed` call in
`scripts/precompute_features.py`. This is a single-line addition with no other
pipeline changes required.

### 4.2 Exact diff

```python
# scripts/precompute_features.py  —  np.savez_compressed call (line ~217)

np.savez_compressed(
    str(out_path),
    features        = features.astype(np.float32),
    labels          = labels.astype(np.int64),
    close           = close.astype(np.float64),
    high            = high.astype(np.float64),
    low             = low.astype(np.float64),
+   tick_volume_raw = df['tick_volume'].to_numpy()[ws:].astype(np.float32),  # VIO feature
    timestamps_ns   = timestamps_ns,
    gmm2            = gmm2.astype(np.float32),
    km_enc          = km_enc.astype(np.float32),
    vol_enc         = vol_enc.astype(np.float32),
    gs_q            = gs_q.astype(np.float32),
    cu_au           = cu_au.astype(np.float32),
    rq              = rq.astype(np.float32),
    atr_norm        = atr_norm.astype(np.float32),
    trend_norm      = trend_norm.astype(np.float32),
    session_phase   = session_phase.astype(np.float32),
    metadata        = np.array(metadata),
)
```

### 4.3 NPZ size impact

`tick_volume_raw` at float32 for N=5,680,771 bars:

```
5,680,771 × 4 bytes = 22.7 MB uncompressed
Estimated compressed (LZ4): ~8–12 MB (tick volume has high run-length compressibility)
Current NPZ size: ~260 MB compressed
Size increase: ~4%
```

### 4.4 Downstream changes after NPZ rebuild

Only `inventory_pressure` computation in the Phase 8 notebook. No changes to:
- `train_supervised.py` (features array unchanged — tick_volume_raw is not in the 10D)
- `train_rl.py` (RL obs vector unchanged)
- `precompute_features.py` feature preparation logic (col 4 tickvol_sc unchanged)

The `tick_volume_raw` key is purely additive — it does not replace or modify any
existing key. All existing code using the NPZ continues to work without modification.

### 4.5 Acceptance criteria

After rebuild, `inventory_pressure_v4` computed via VIO formula should satisfy:
- `|skew| < 1.0`
- `kurt ∈ [−2, +5]`
- `std > 0.10` (meaningful variance — not zero-inflated)
- ACF decays within W bars (W=20) with clear temporal structure

If any criterion is not met, the tick_volume data itself has anomalies
(MT5 broker-reported volume vs actual exchange volume) — escalate to data source review.

---

## 5. Phase 8 Candidate Set — Confirmed

Three features proceed to Phase 8 (full KS/MI/redundancy label-based validation):

| Feature | Theoretical basis | Max ext \|r\| | ACF decay | Phase 8 action |
|---------|-----------------|--------------|-----------|----------------|
| `adverse_selection_proxy` | Hagströmer §2.3 permanent impact | 0.278 | ~10 bars | Validate vs existing 10D |
| `order_processing_residual` | Hagströmer §2.5 spread residual | 0.087 | ~15 bars | Validate — highest priority |
| `hawkes_excitation_5` | El Euch τ=5 (τ=96 variant also) | 0.208 | ~5 bars | Validate; add τ=96 comparison |
| `inventory_pressure_v4` | Hagströmer §2.4 VIO | — | — | **Blocked** pending NPZ rebuild |

**Phase 8 pass thresholds** (from Phase 4 + Phase 6 baseline):

| Criterion | Threshold | Best existing feature |
|-----------|-----------|----------------------|
| KS(sell vs hold) | D > 0.05 | `spread_pressure` D=0.600 |
| Mutual information | MI > 0.005 | `spread_pressure` MI=0.056 |
| Redundancy ratio | < 0.30 | — |

Any candidate with KS > 0.15 AND MI > 0.010 AND redundancy < 0.30 would rank in the
top 3 of the full feature set history (Phases 4, 6, 7).

---

## 6. Theoretical Reconciliation — What Phase 7 Established

### 6.1 Hawkes / rough volatility at M1 resolution

El Euch et al.'s H≈0.10 and Hawkes kernel decay τ≈40 bars are derived at tick or
sub-second frequency. At M1 resolution, each bar aggregates ~60 individual tick events.
The self-exciting dynamics visible at tick frequency are averaged out at M1.

**Empirical results confirm this:**
- H(log_ret) = 0.513 — Brownian, consistent with market efficiency at 1-minute aggregation
- H(log_RV) = 0.834 — volatility persistence (GARCH effect), not rough volatility
- Empirical Hawkes τ = 96 bars at M1 — reflects macro-level momentum, not tick-level excitation

The Hawkes excitation framework remains valid as a *motivation* for directional momentum
features at M1 (the `hawkes_excitation_5` and `hawkes_excitation_96` features), but the
theoretical parameter values (τ, H) do not transfer from tick to M1 resolution.

### 6.2 Hagströmer spread decomposition — validated and productive

The 50%/25%/25% decomposition of gold futures bid-ask spread (order processing /
adverse selection / inventory) has produced two successful Phase 8 candidates.
`order_processing_residual`'s near-orthogonality to the existing 10D (max |r|=0.087)
confirms the decomposition extracts independent information.

**The key limitation** is data availability: the inventory component requires raw
tick_volume (not the scaled proxy in col 4 or the z-scored version in col 8).
This is the precompute pipeline change requested in §4.

### 6.3 Regime structure

All Phase 7 features are regime-stable (KS Bear vs Bull < 0.10 for all candidates).
This is a feature property, not a failure — these are short-memory, bar-level signals
that do not carry regime-level information. Regime context is handled by the 6D regime
array in the RL observation space. The supervised model receives these features as
per-bar microstructure signals; regime conditioning is the RL agent's responsibility.

---

## 7. Priority Checklist

### Immediate (Phase 8 unblocked)
- [ ] Copy `phase7v2_signal_quality_lab.ipynb` to repo `notebooks/`
- [ ] Run Phase 8 notebook on 3 confirmed candidates with labels
- [ ] Add `hawkes_excitation_96` to Phase 8 candidate set (τ=96 from corrected empirical τ)
- [ ] Continue RL training in parallel (independent of feature exploration track)

### Deferred (after RL convergence)
- [ ] Apply `tick_volume_raw` patch to `precompute_features.py` (§4.2 diff)
- [ ] Rebuild `training_ready.npz` with new key
- [ ] Add `inventory_pressure_v4` (VIO formula, §3.4) to Phase 8+ notebook
- [ ] Validate VIO against kurtosis acceptance criteria (§4.5) before Phase 9

### Architecture note for master agent
The `precompute_features.py` change is low-risk and purely additive. It should be
applied at the next scheduled NPZ rebuild (which will occur anyway when Phase 8
adds new features to the supervised branch). There is no reason to rebuild the NPZ
solely for this change before Phase 8 completes.

---

*Architecture Decision Record — 2026-05-05*
*HFTExperiment v2, Phase 7 → Phase 8 (Feature Signal Quality Lab)*
