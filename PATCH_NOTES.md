# HFTExperiment — Patch Notes

## v2 — Architecture Improvements (current branch)

Based on review of 10 papers (see IMPROVEMENTS.md) and supervised/RL training
results (see TRAINING_REPORT.md).

### Priority 1 — Confidence calibrated against realised Sharpe (§5.1)
**File:** `src/training/confidence_calibration.py` (new)

The existing confidence head was trained as MSE(is_correct) against hold-dominated
batches — by ep18 it output 0.974 with sell_recall=0.079 (confidently wrong).
Activation variance is 96% uncorrelated with output importance (Variance≠Importance).

Changes:
- `SharpeConfidenceHead`: regresses next-20-bar realised Sharpe from intermediate encoder layer (60-70% depth per §5.2)
- `IsotonicCalibrator`: post-hoc isotonic regression on validation split
- Integration point: `src/encoder/fusion.py` → tap intermediate layer for confidence

### Priority 2 — Transaction cost curriculum (§3.2)
**File:** `scripts/train_rl.py` (rewritten)

CCSO §3.2: anneal fees from 0 → real over N evolves. Different evolve-stage models
behave qualitatively differently; deploy by volatility regime.

v3 RL diagnosis (root cause): signal entry bonus/penalty (-1.0/+0.5) at 99 trades/ep
dominated the reward signal, causing penalty-avoidance not PnL optimisation.
Removed. Cost curriculum achieves the same goal at the correct reward scale.

Changes:
- `CostCurriculum` class: `commission: 0 → $0.70`, `spread: 0 → 2 pip` over N_EVOLVES stages
- Checkpoint saved at each evolve boundary (`rl_agent_evolve{N}.pt`)
- Deployment: route by volatility regime (Bear+HIGH → high evolve, conservative)
- Signal entry bonus/penalty removed
- `episode_len=8000` retained (99 trades/ep = 4× learning signal vs 2k)

### Priority 3 — Encoder: CCSO two-stage architecture (§1.2, §1.3)
**File:** `src/encoder/price_branch.py` (rewritten)

CCSO: inter-sequence (cross-feature) Conv first, then intra-sequence (temporal)
LocalAttention. Performance improves then degrades past w≈20 bars (their Fig 3).

Changes:
- `CrossFeatureConv`: Linear(F→2H) + GELU + Linear → mixes feature axis first
- `LocalCausalAttention`: sliding causal mask w=20; O(n·w²) not O(n²)
- Two-stage: stage1 → stage2 × n_layers

### New files

| File | Purpose | Status |
|------|---------|--------|
| `src/encoder/price_branch_sessa.py` | Sessa mixer A/B vs CCSO branch | Stub — wired to LocalCausalAttention |
| `src/meta_policy/lewm_world.py` | LeWM JEPA world model (GAN replacement) | SIGReg implemented; predictor MLP stub |
| `src/meta_policy/regime_router.py` | Delay-line buffer + cross-attn router (HD-SNN §7) | Implemented |
| `src/training/confidence_calibration.py` | Sharpe regression + isotonic calibration (§5.1) | Implemented |

### Changes to existing files

| File | Change |
|------|--------|
| `src/encoder/price_branch.py` | Full rewrite: CCSO two-stage |
| `scripts/train_rl.py` | Full rewrite: cost curriculum, remove signal penalty |
| `src/data/preprocessing.py` | Added `compute_rl_obs_features` (merged from feature_engineering) |
| `src/training/train_supervised.py` | v4: FocalLoss, ReduceLROnPlateau, regime-stratified split, F1 criterion |

### Things NOT done (per §10 "don't do" list)
- No SVD/rank-reduction compression — INT4 quantisation dominates 2–25× MSE
- No multi-block linear approximation — distribution shift cascades through residuals
- No PCA rotation of inputs — 20× worse MSE than direct INT4
- No unbounded attention window growth — CCSO Fig 3 shows degradation past regime pattern length

---

## Phase 3 Regime-Informed Pipeline (prior)

See Phase 3 branch for:
- join_regime_labels(), get_regime_array(), regime-stratified split
- GMM2 Bear entry gate, G/S-conditioned max_hold
- SequenceDataset OOM fix
- 5 supervised runs, 3 RL runs
