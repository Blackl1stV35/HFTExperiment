# PATCH_NOTES_PHASE8_VERDICT_06052026.md

**Date:** 2026-05-06  
**Release tag:** pre-release-v0.5.4.0-phase8-verdict  
**Authors:** Blackl1stV35 + Claude (Anthropic)  
**Status:** PENDING — Phase 8 label-based validation not yet run

---

## Context

This note documents the Phase 8 candidate set derived from Phase 7v2 results
(`pre-release-v0.5.3.2-beta`), the `inventory_pressure` pipeline patch required
before Phase 8 can run, and the architectural decisions that follow from the
Phase 6 + Phase 7 combined findings.

---

## 1. Phase 8 Candidate Set

Three microstructure features are cleared for Phase 8 label-based validation.
Two are permanently decided. Status summary:

| Feature | Status | Action |
|---------|--------|--------|
| `adverse_selection_proxy` | ✅ READY | Include in Phase 8 run |
| `order_processing_residual` | ✅ READY | Include in Phase 8 run (priority — max ext\|r\|=0.087, most orthogonal) |
| `hawkes_excitation_5` | ✅ READY | Include in Phase 8 run |
| `hawkes_excitation_96` | ✅ READY (not yet computed) | Add to Phase 8 before running — captures full τ=96 empirical kernel support |
| `inventory_pressure` | ❌ BLOCKED | Requires `tick_volume_raw` in NPZ — see §3 |
| `roughness_rv` | ❌ PERMANENTLY EXCLUDED | Measures GARCH volatility clustering (H≈0.79), not rough volatility (H≈0.10). El Euch et al. framework inaccessible at M1 resolution via R/S or DFA. No further work. |

### Phase 8 protocol

Phase 8 is label-based validation of the READY candidates using the same
KS / MI / redundancy + temporal profiling methodology as Phase 6, but against
the 3-class ATR-adaptive label set (`training_ready.npz`). Pass criteria:

- **KS D > 0.05** (sell vs hold)
- **MI > 0.005** (above OHLCV baseline mean)
- **Redundancy ratio < 0.30** (max pairwise MI / label MI)
- **Temporal locality**: Cohen's d profile confirms stream assignment (short w=20 vs long T=120)

All four READY candidates (including `hawkes_excitation_96`) must be computed
and validated in a single notebook run. Notebook: `phase8_label_validation.ipynb`.

If ≥1 candidate passes all 4 criteria → supervised NPZ rebuild required:
- `n_bypass_features` in `dual_branch.yaml` set to number of passing features
- `compute_bypass_features()` in `precompute_features.py` already implements
  `adverse_selection_proxy`, `order_processing_residual`, `hawkes_excitation_5`
  (add `hawkes_excitation_96` stub before running)
- NPZ renamed `training_ready_phase8_Nfeat.npz` to prevent stale loads
- Supervised training restarted from scratch (input dimensionality change)

If 0 candidates pass → 10D NPZ is permanent ceiling for supervised model.
Proceed to extended RL training only.

---

## 2. hawkes_excitation_96 — Compute Spec

τ=96 was identified as the corrected empirical Hawkes kernel decay constant
from Phase 7v2 (consecutive-segment ACF on 100,000 bars). Compute identically
to `hawkes_excitation_5`, substituting `tau=96`:

```python
def hawkes_excitation(log_abs_ret: np.ndarray, tau: int) -> np.ndarray:
    """Exponentially-weighted self-excitation intensity.

    I[t] = sum_{s < t} |r_s| * exp(-(t-s) / tau)
    Implemented as causal IIR filter: I[t] = I[t-1]*exp(-1/tau) + |r[t]|
    """
    alpha = np.exp(-1.0 / tau)
    n     = len(log_abs_ret)
    intensity = np.empty(n, dtype=np.float64)
    intensity[0] = abs(log_abs_ret[0])
    for i in range(1, n):
        intensity[i] = intensity[i-1] * alpha + abs(log_abs_ret[i])
    return rolling_zscore(intensity, window=120)
```

Add to `phase8_label_validation.ipynb` §2 (feature computation) before running.
Both τ=5 and τ=96 should be tested — they capture different timescales of
self-excitation (fast microstructure vs macro momentum persistence).

---

## 3. inventory_pressure — Pipeline Patch Required

### Root cause (confirmed Phase 7v2 §9)

All four `inventory_pressure` variants (v1–v4) fail with pathological kurtosis
(>100) due to zero-inflation in `vol_zscore` (NPZ column 8). XAUUSD M1 tick
volume is near-constant during Asian session hours and weekly open/close
boundaries. Near-constant input → rolling z-score ≈ 0 → tanh(0) = 0. The
rare London-open and macro spikes dominate the 4th moment, driving kurtosis
toward infinity as denominator variance → 0.

### Correct fix: Volume Imbalance Order (VIO) formula

The fix requires **raw unscaled tick volume** — not the current `tickvol_sc`
(MinMax-scaled, col 4) or `vol_zscore` (tanh-compressed, col 8).

**VIO formula:**
```
VIO[t] = Σ_{k=t-W+1}^{t} (V_k × sign_k) / Σ_{k=t-W+1}^{t} V_k
```

Where `V_k` is raw tick count per bar (integer, unscaled) and
`sign_k = sign(close_k - open_k)` (trade direction). Dividing by total
rolling volume normalises session-level magnitude differences. The ratio is
bounded to (-1, 1) with no zero-inflation.

### Pipeline change required

**`scripts/precompute_features.py`** — add `tick_volume_raw` extraction and
save to NPZ:

```python
# In the NPZ save block, add:
'tick_volume_raw': tick_volume_raw.astype(np.int32),   # raw bar tick count
```

Where `tick_volume_raw` is the integer tick count from the source DuckDB
before any scaling. It is already present in the raw DataFrame as
`df['tick_volume']` (or equivalent column name from TickStore).

**`training_ready.npz` rebuild required** after this patch. Rename output
to `training_ready_vio.npz` to avoid stale loads.

**After NPZ rebuild**, validate `inventory_pressure_vio` in
`phase8_label_validation.ipynb` using the same KS/MI/redundancy criteria.
If it passes, it becomes a Phase 8 supervised candidate alongside the
three READY features.

### Estimated effort

1. Identify tick_volume column name in TickStore output — `df.columns` check
2. Add to NPZ save dict in `precompute_features.py` — 2 lines
3. Rebuild NPZ — ~10 minutes on A100 (precompute step)
4. Add VIO computation to Phase 8 notebook — ~20 lines
5. Run validation

---

## 4. Architectural State After Phase 6 + Phase 7

### What changed

| Component | State |
|-----------|-------|
| `training_ready.npz` | Unchanged — 10D features, no rebuild |
| `price_branch_transformer.py` | `n_bypass=0` wired (no-op) — bypass arch ready for Phase 8 if needed |
| `dual_branch.yaml` | `n_bypass_features: 0` — change after Phase 8 if features pass |
| `train_rl.py` | **Extended to 15-dim obs** — `ret_1h` (obs[13]) + `ret_15m` (obs[14]) added |
| `precompute_features.py` | `compute_bypass_features()` stub ready — add `hawkes_excitation_96` before Phase 8 |
| Supervised model | Unchanged — Run 8 best checkpoint (`dual_branch_best.pt`, ep61) |

### What the Phase 6 finding confirms about the supervised ceiling

Phase 6's strongest candidate (`ret_1h`, KS=0.361) is redundant with the existing
10D set (ratio=1.033) — meaning the Transformer at d_model=512, 4 layers, already
implicitly computes the equivalent of a 60-bar cumulative return through attention
over the `bar_return_bps` sequence. The information is present. The supervised
precision ceiling (sell P=0.299 test, Run 8) is not a feature coverage problem.
It is a combination of:

1. **Label noise** — ATR-adaptive barriers produce borderline labels near TP/SL
   boundary; structurally unpredictable regardless of features
2. **Class imbalance dynamics** — sell 3.48%, buy 0.76%; regime-stratified
   sampler may still under-represent Bear+HIGH sequences
3. **RL headroom** — confidence gating and position sizing at the RL layer can
   recover precision the flat 3-class classifier cannot express

Phase 7's microstructure features (`adverse_selection_proxy`,
`order_processing_residual`, `hawkes_excitation_5/96`) are genuinely orthogonal
(max ext|r|=0.087–0.278) and may push past the ceiling if Phase 8 validates them.
They measure information the Transformer does not implicitly compute from OHLCV —
namely, the permanent vs transitory decomposition of price impact and the
self-excitation intensity of the return process.

---

## 5. Immediate Action List

| Priority | Action | Blocker |
|----------|--------|---------|
| **NOW** | Run RL Phase 4 with 15-dim obs (`train_rl.py` patched) | None — unblocked |
| **SOON** | Add `tick_volume_raw` to NPZ, rebuild | Manual pipeline patch |
| **SOON** | Compute `hawkes_excitation_96`, add to Phase 8 notebook | Requires NPZ or inline computation |
| **SOON** | Run `phase8_label_validation.ipynb` on 4 READY candidates | Requires Colab + NPZ |
| **DEFERRED** | If Phase 8 passes ≥1 feature: set `n_bypass_features`, retrain supervised | Phase 8 results |
| **PERMANENT** | No further roughness/Hurst work at M1 resolution | — |

---

## 6. RL Phase 4 Resume Command

Use Run 8 best checkpoint (`dual_branch_best.pt`, ep61, val signal=0.171,
val sell P=0.193, test sell P=0.299):

```bash
python scripts/train_rl.py \
    --checkpoint "/content/drive/MyDrive/Colab Notebooks/dual_branch_best.pt" \
    --steps 1500000 \
    --n-evolves 10 \
    --episode-len 8000 \
    --confidence-gate 0.70 \
    --commission 0.70 \
    --spread-pips 2.0 \
    --eval-every 16000 \
    --n-eval-episodes 15 \
    --curriculum-warmup 100000 \
    --save-dir "/content/drive/MyDrive/Colab Notebooks/rl_checkpoints"
```

Obs vector is now 15-dim. Agent network is `ConfidenceSACAgent(obs_dim=15)` —
incompatible with previous 13-dim checkpoints. Start from scratch (no resume
from Phase 3 RL checkpoints). The improved supervised backbone (sell P=0.299
vs 0.283 in Phase 3) should raise the RL WR ceiling above the previous 49.6%.

---

*HFTExperiment v2 · Phase 8 pending · Next checkpoint: RL Phase 4 results*
