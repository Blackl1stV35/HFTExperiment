# XAUUSD Dual-Branch Supervised + SAC RL — Training Report

**Project:** HFTExperiment Phase 3 (Regime-Informed Pipeline)
**Period:** 2026-04-19 → 2026-04-25
**Dataset:** XAUUSD M1, 5,998,591 bars (2005–2026), ~6,666 days
**Hardware:** Google Colab Pro — Tesla T4 (16 GB VRAM)

---

## Supervised Training — 5 Runs

### Dataset facts
| Item | Value |
|------|-------|
| Total sequences | 5,680,771 |
| Sell labels | 116,994 (2.1%) |
| Hold labels | 5,540,842 (97.5%) |
| Buy labels | 22,935 (0.4%) |
| Val set n_sell | 12,179 |
| Val set n_buy | 1,411 |
| Epoch size (sampler) | 500,000 sequences |
| Epoch wall-clock | ~11 min on T4 |

### Run history

| Run | Key config | Best checkpoint | Criterion | Test sell F1 | Test buy R | Notes |
|-----|-----------|----------------|-----------|-------------|-----------|-------|
| 1 | CE loss, clip=2, LR=2e-4 | ep3, val_loss=0.429 | val_loss ↓ | 0.066 | 0.010 | OOM crash ep4; 16.4 GB X array |
| 2 | clip=5, LR=1e-4, post-2024 ×0.5 | ep4, score=0.262 | signal_score | — | 0.010 | Colab timeout 97 min |
| 3 | clip=10, LR=5e-5 | ep5, score=0.282 | signal_score | — | 0.011 | GradNorm still hitting 10.0 ceiling |
| 4 | Focal γ=2, buy=10 | ep14, score=0.535 | signal_score | sell P=0.160 | 0.021 | Recall-only metric saved wrong checkpoint; val_acc oscillated 0.10–0.54 |
| **5** | **Focal γ=1, buy=5, F1 criterion, ReduceLROnPlateau** | **ep60, score=0.032** | **sell+buy F1** | **sell P=0.253 R=0.611 F1=0.358** | **0.018** | **Best overall; 60 epochs completed** |

### Run 5 details (accepted model)

**Config changes from run 4:**
- `focal_gamma: 1.0` (was 2.0) — hold weight at γ=1 is 0.02, not 0.0004
- `class_weights: [2.5, 0.3, 5.0]` (was 10.0 for buy) — 5× proportional to label ratio
- `signal_score` → F1-based: `sell_F1×0.4 + buy_F1×0.4` — penalises precision=0
- `CosineAnnealingWarmRestarts` → `ReduceLROnPlateau(mode=max, patience=5, factor=0.5)`
- Regime-stratified split: each GMM2×vol bucket contributes proportionally to all splits
- Start fresh (no resume from run 4 recall-only checkpoint)

**Key epoch milestones:**

| Phase | Epochs | Pattern |
|-------|--------|---------|
| 1–20 | Focal oscillation | Val acc 0.09–0.54, model swings minority/hold dominant |
| 21–30 | Transition | Model stabilises to hold-dominant, sell recall falls 0.42→0.32 |
| 31–60 | Convergence | Train acc 0.83→0.91, sell P creeps 0.021→0.023, LR decays 5e-5→1.25e-5 |

**Test result (ep60 checkpoint):**

| Metric | Value |
|--------|-------|
| Test loss | 2.895 |
| Test acc | 0.695 |
| Sell precision | 0.253 |
| Sell recall | 0.611 |
| Sell F1 | 0.358 |
| Buy precision | 0.025 |
| Buy recall | 0.018 |
| Signal score | 0.151 |

**Key finding:** Sell signal generalises (val→test recall improves 0.23→0.61). Buy recall collapses on test (0.44→0.018) across all 5 runs — structural temporal distribution mismatch; val buy examples are from a different regime than test.

### Infrastructure fixes across runs

| Fix | Run introduced | Impact |
|-----|---------------|--------|
| `SequenceDataset` on-the-fly slicing | Run 2 | RAM: 16.4 GB → 0.13 GB; no more OOM |
| `WeightedRandomSampler(num_samples=500k)` | Run 1 | Epoch size controlled; ~11 min/epoch |
| Google Drive mirror in `save_checkpoint()` | Run 2 | Checkpoint survives Colab disconnection |
| Signal score checkpoint criterion | Run 2 | No longer saves underfitting epoch |
| F1-based signal score | Run 5 | Prevents precision=0 checkpoint |
| Post-2024 undersample removed | Run 3 | Buy recall on test improved slightly |
| Regime-stratified split | Run 5 | Val regime composition matches test |

---

## RL Training — 3 Runs

### Supervised model used for all RL runs
Frozen at run 5 checkpoint (ep60). Signal distribution: sell=13.2%, hold=84.0%, buy=2.8%. Confidence mean=0.908±0.224.

### Run history

| Run | episode_len | steps | best_eval | mean_eval | eval_std | mean_WR | trades/ep |
|-----|-------------|-------|-----------|-----------|----------|---------|-----------|
| RL-1 | 2,000 | 500k | $2,065 | −$121 | $381 | 44.2% | 25 |
| RL-2 | 2,000 | 1,500k | $413 | −$121 | $125 | 44.2% | 25 |
| RL-3 | 8,000 | 1,500k | $172 | −$520 | $244 | 44.0% | 99 |

### RL-1 analysis
- 3-episode eval had std=$381 — single lucky window caused $2,065 best (step 64k, WR=38%)
- Train PnL diverged to −$190 after step 240k: replay buffer filled with losing transitions
- confidence_gate=0.48 was inactive (effective scale 0.82 on every trade)

### RL-2 key changes and outcomes
- n_eval_episodes: 3→15 (std fell 3× to $125 as predicted)
- confidence_gate: 0.48→0.70 (meaningful differentiation)
- buy_reward_scale=0.30 (buy F1=0.021 on test)
- post-2024 undersample removed
- Result: train PnL flat at −$85 to −$110 across all 1.5M steps. WR 44% all phases. 11% positive evals.

### RL-3 key changes and outcomes
- episode_len: 2,000→8,000 (trades/ep: 25→99 — 4× more learning signal)
- Signal entry bonus (+0.50) and penalty (−1.00) at entry bar
- eval_every: 8k→16k
- Result: trades/ep increased correctly to 99. But eval mean fell to −$520. Only 2.2% positive evals.

### RL-3 root cause diagnosis

**Problem 1 — Signal penalty scale mismatch.** Penalty (−1.00/entry) vs MTM reward (~$0.05/bar). At 99 trades/ep, net shaping component was +$36/episode — penalty was suppressing entries (penalty-avoidance policy) rather than improving signal alignment.

**Problem 2 — Transaction cost friction.** $0.90/round-trip × 99 trades = $89/episode in pure costs. ~28% of total losses were transaction friction before any market P&L.

**Problem 3 — Supervised sell precision ceiling.** Breakeven WR ≈ 50–51% at $0.90/trade. Agent WR pinned at 44% across all three RL runs regardless of hyperparameters. With sell P=0.253, 75% of sell predictions are hold misclassified — these trade as losses. No reward shaping can overcome a negative-expectancy signal at realistic costs.

### Conclusion

The supervised model (sell F1=0.358, P=0.253) is not sufficient to drive a profitable RL strategy at $0.90/round-trip transaction costs. Two paths forward:

1. **Improve supervised model precision** to ≥0.35 via architecture changes (CCSO two-stage encoder, Sessa mixer)
2. **Train RL in a zero-cost curriculum environment** (CCSO §3.2): start commission=0, anneal to real values over evolves. Agent learns the signal edge first, then cost management.

Both are addressed in the v2 architecture (see `IMPROVEMENTS.md` and `PATCH_NOTES.md`).

---

## Regime Research (Phase 3 context)

**Current macro (2026-04-22):** GMM2=Bear (P(exit today)=0.7%), vol=HIGH (19.9%), G/S=Q1 (rank 0.15), Cu-Au=Commodity. Bear+HIGH = Sharpe 0.33 (worst reliable cell). Deploy at reduced size until vol crosses below 14.3% threshold.
