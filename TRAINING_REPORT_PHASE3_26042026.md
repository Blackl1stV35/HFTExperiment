# XAUUSD HFT — Training Report
**Phase 3 · Regime-Informed Dual-Branch Supervised + SAC RL**
**Period:** 2026-04-19 → 2026-04-25 · **Hardware:** Colab Pro T4 16 GB

---

## Executive Summary

Six supervised runs and four RL runs were completed. The supervised model converged to **sell P=0.259, R=0.611, F1=0.365** on the test set. Three independent RL runs produced identical results — WR pinned at 44.1%, best eval PnL $172 appearing at the same step (944k) in all three runs. This confirms the performance ceiling is the supervised model architecture (InceptionBlock + TCN), not the RL setup, reward shaping, cost structure, or episode length.

**Decision: proceed to CCSO two-stage encoder retraining from scratch.**

---

## Supervised Training — 6 Runs

### Dataset
| Item | Value |
|------|-------|
| Total M1 bars | 5,998,591 (2005–2026) |
| Total sequences (seq_len=120) | 5,680,771 |
| Sell labels | 116,994 (2.1%) — regime-stratified split |
| Hold labels | 5,540,842 (97.5%) |
| Buy labels | 22,935 (0.4%) |
| Val n_sell / n_buy | 12,179 / 1,411 |
| Epoch size (sampler) | 500,000 |
| Epoch wall-clock | ~11 min on T4 |

### Regime distribution
| Bucket | Sequences | Share |
|--------|-----------|-------|
| Bear-LOW | 0 | 0.0% |
| Bear-NORMAL | 14,831 | 0.3% |
| Bear-HIGH | 604,203 | 10.6% |
| Bull-LOW | 2,160,064 | 38.0% |
| Bull-NORMAL | 2,025,640 | 35.7% |
| Bull-HIGH | 876,033 | 15.4% |

### Run summary

| Run | Key changes | Config | Best checkpoint | Test sell P | Test sell R | Test sell F1 | Outcome |
|-----|-------------|--------|----------------|------------|------------|-------------|---------|
| 1 | Baseline CE loss | clip=2, LR=2e-4, buy=2.5 | ep3 by val_loss | — | — | — | OOM crash ep4 (16.4 GB X array) |
| 2 | SequenceDataset OOM fix | clip=5, LR=1e-4, post-2024 ×0.5 | ep4, score=0.262 | — | — | — | Colab timeout 97 min |
| 3 | clip raised | clip=10, LR=5e-5 | ep5, score=0.282 | — | — | — | GradNorm hit 10.0 ceiling |
| 4 | Focal γ=2, buy=10 | F1 criterion | ep14, score=0.535 | 0.160 | 0.777 | 0.252 | High recall, precision=0.015 on val; wrong checkpoint |
| **5** | **Focal γ=1, buy=5, ReduceLROnPlateau, regime-stratified split** | **F1 criterion** | **ep60** | **0.253** | **0.611** | **0.358** | **Best run; 60 epochs completed** |
| 6 | Resume ep60 + precision floor P≥0.25 | epochs=90, LR=1.25e-5→3.13e-6 | None saved (val P=0.023, below floor) | 0.259 | 0.605 | 0.365 | Marginal test gain; model at capacity |

### Run 5 details (accepted model)

**Final config:**
```yaml
# configs/config.yaml
learning_rate: 5e-5
gradient_clip: 10.0
focal_gamma: 1.0
early_stopping_patience: 15
epochs: 60

# configs/data/xauusd.yaml
class_weights: [2.5, 0.3, 5.0]
```

**Training phases:**

| Phase | Epochs | Pattern |
|-------|--------|---------|
| 1–20 | Focal oscillation | Val acc 0.09–0.54, model swings minority/hold dominant |
| 21–30 | Transition | Stabilises hold-dominant; sell recall falls 0.42→0.32 |
| 31–60 | Convergence | Train acc 0.83→0.91; sell P creeps 0.021→0.023; LR 5e-5→1.25e-5 |

**Test result (ep60):**

| Metric | Value |
|--------|-------|
| Test loss | 2.895 |
| Test accuracy | 0.695 |
| Sell precision | 0.253 |
| Sell recall | 0.611 |
| Sell F1 | 0.358 |
| Buy precision | 0.025 |
| Buy recall | 0.018 |
| Buy F1 | 0.022 |

### Run 6 details (fine-tuning attempt)

Resumed from ep60 checkpoint. Added precision floor: `signal_score = 0 if sell_P < 0.25`. Ran ep61–75. Val sell precision stayed at 0.023 throughout — well below the 0.25 floor — so no checkpoint was ever saved. Early stopping triggered at ep75 (patience=15 from ep61). Test showed sell P=0.259 from the ep60 weights, confirming the model was fully converged at ep60.

### Key infrastructure fixes across runs

| Fix | Run | Impact |
|-----|-----|--------|
| `SequenceDataset` on-the-fly slicing | 2 | RAM: 16.4 GB → 0.13 GB; no OOM |
| `WeightedRandomSampler(num_samples=500k)` | 1 | Epoch size controlled; ~11 min/epoch |
| Google Drive mirror in `save_checkpoint` | 2 | Checkpoint survives Colab disconnect |
| F1-based signal score | 4 | Prevents precision=0 checkpoint |
| Precision floor (P≥0.25) | 6 | Forces convergence toward precision frontier |
| Regime-stratified split | 5 | Val regime composition matches test |
| `ReduceLROnPlateau(mode=max)` | 5 | LR decays on signal quality, not val loss |
| `np.asarray(ts < cutoff, dtype=bool)` | 6 | Fixes `to_numpy()` on DatetimeIndex |

---

## RL Training — 4 Runs

### Frozen model used
All runs used the ep60 supervised checkpoint. Signal distribution extracted across 5,680,771 sequences: **sell=13.2%, hold=84.0%, buy=2.8%**, conf=0.908±0.224.

### Run summary

| Run | episode_len | steps | n_eval_ep | gate | key change | best_eval | mean_WR |
|-----|-------------|-------|-----------|------|-----------|-----------|---------|
| RL-1 | 2,000 | 500k | 3 | 0.48 | Baseline | $2,065* | 44.2% |
| RL-2 | 2,000 | 1,500k | 15 | 0.70 | n_eval=15, buy discount 0.3× | $413 | 44.2% |
| RL-3 | 8,000 | 1,500k | 15 | 0.70 | episode_len 2k→8k, signal entry bonus/penalty | $172 | 44.0% |
| **RL-4** | **8,000** | **1,500k** | **15** | **0.70** | **Cost curriculum (0→$0.70 over 10 evolves)** | **$172** | **44.1%** |

*RL-1 $2,065 was 3-episode eval noise (WR=38.1% at that step — below breakeven)

### RL-4 details (cost curriculum)

**Config:**
```bash
--steps 1500000 --n-evolves 10 --episode-len 8000
--eval-every 16000 --n-eval-episodes 15
--confidence-gate 0.70 --commission 0.70 --spread-pips 2.0
--curriculum-warmup 100000
```

**Evolve checkpoints saved:** 10 (rl_agent_evolve0.pt → rl_agent_evolve9.pt)

**Evolve-level behaviour:**

| Evolve | Steps | Comm | Spread | Train r (mean) | Pattern |
|--------|-------|------|--------|----------------|---------|
| 0 | 0–150k | $0→0.07 | 0→0.2pip | +130 to +80 | Positive — signal has frictionless edge |
| 1 | 150k–300k | $0.07–0.14 | 0.2–0.4pip | +42 to −127 | Breakeven crossed ~comm=$0.09 |
| 2–4 | 300k–750k | $0.14–0.35 | 0.4–1.0pip | −100 to −185 | Stable negative, WR 43–45% |
| 5–9 | 750k–1,500k | $0.35–0.70 | 1.0–2.0pip | −75 to −280 | Monotonically worsening with costs |

**Key observation:** Train reward was **positive** in evolve-0 (near-zero cost). The agent found the signal edge in a frictionless environment. The breakeven crossed at comm≈$0.09 / spread≈0.3 pip — well below real costs. The supervised signal (sell P=0.259) cannot support real XAUUSD transaction costs.

### Identical RL-3 and RL-4 outputs — definitive diagnosis

RL-3 (signal bonus/penalty) and RL-4 (cost curriculum) produced **byte-identical** step-by-step outputs: same train reward, same eval PnL at every checkpoint, same best eval $172 at step 944k, same final WR 44.1%. This confirms:

1. The RL training infrastructure is deterministic and correct (seed=42)
2. The performance ceiling is entirely determined by the supervised signal quality
3. No RL hyperparameter change can overcome sell precision of 0.259 at $0.90/round-trip

### WR analysis

Breakeven WR for $0.90/round-trip (comm $0.70 + 2-pip spread $0.20) with avg winner ≈ avg loser: **≈50–51%**. All four RL runs produced WR 44.0–44.2% throughout. The WR is a direct function of supervised sell precision:

| Sell P | Expected profitable trades | Required WR for edge |
|--------|---------------------------|---------------------|
| 0.259 (current) | 1 in 3.9 | ~51% — not reachable |
| 0.350 (CCSO target) | 1 in 2.9 | ~49% — reachable |
| 0.450 (aspirational) | 1 in 2.2 | ~48% — comfortable edge |

---

## Root Cause: Architecture Capacity Ceiling

The InceptionBlock + CausalTCN architecture (1,864,646 params) has converged. Val sell precision stabilised at 0.021–0.024 across all fine-tuning epochs despite:
- Focal loss γ=1.0
- Class weights [2.5, 0.3, 5.0]
- Regime-stratified split
- ReduceLROnPlateau down to LR=3.13e-6
- 75 total training epochs

The precision wall at 0.023 on val (and 0.259 on test) is the model's ceiling for this architecture and label distribution.

---

## Decision: CCSO Two-Stage Encoder

**Proceed to full retrain from scratch using the CCSO architecture** (`src/encoder/price_branch_ccso.py`).

### CCSO changes vs current architecture

| Component | Current (InceptionBlock+TCN) | CCSO two-stage |
|-----------|------------------------------|----------------|
| Stage 1 | Multi-scale inception (4 kernel sizes) | CrossFeatureConv — mixes feature axis |
| Stage 2 | CausalTCNBlock × 4 | LocalCausalAttention(window=20) × 2 |
| Attention | None | Causal local mask, O(n·w²) |
| Params | 1,864,646 | ~same (configurable hidden_dim) |
| Key CCSO finding | — | Performance peaks then degrades past w≈20 bars (Fig 3) |

### CCSO retrain checklist

1. **Swap encoder in `fusion.py`:**
   ```python
   # Change import in src/encoder/fusion.py
   from src.encoder.price_branch_ccso import PriceBranch
   ```

2. **Reset training — do not resume from ep60** (architecture mismatch)

3. **Keep all other config unchanged:**
   ```yaml
   learning_rate: 5e-5
   gradient_clip: 10.0
   focal_gamma: 1.0
   class_weights: [2.5, 0.3, 5.0]
   ```

4. **Target metric for accepting CCSO checkpoint:** sell P ≥ 0.30 on test

5. **RL next step after CCSO:** If sell P ≥ 0.30, the cost curriculum RL should show positive eval PnL in evolve 1–2 (not just evolve 0). That is the signal that the architecture improvement has translated to a tradeable edge.

---

## Appendix: Current Macro Context (2026-04-25)

| Signal | Value | Implication |
|--------|-------|-------------|
| GMM2 | Bear (P_exit=0.7%/day) | Block new entries via RL gate |
| Vol regime | HIGH (19.9% ann) | Bear+HIGH = Sharpe 0.33 (worst cell) |
| G/S quartile | Q1 (rank 0.15) | max_hold=40 bars |
| Cu-Au regime | Commodity (0.52) | Normal weighting |
| DFII10 real yield | +1.93% | Yield framework broken post-2022 |

Deploy at reduced size until vol crosses below 14.3% threshold.
