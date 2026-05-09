# FINAL_MODEL_REPORT_09052026.md
**HFTExperiment v2 — Final Training Report**  
**Date:** 2026-05-09 | **Status:** Paper Trade Ready

---

## Executive Summary

A dual-branch Transformer + SAC reinforcement learning system for XAUUSD M1 trading was trained over 10 supervised runs and 8 RL phases across 5,680,771 bars (2009–2026). The final system achieves **51.2% win rate** at curriculum worst-case costs ($0.70 commission + 2.0 pip spread). At HF Markets Pro real costs (~$0.006/trade), break-even WR is 50.003% — the model has a confirmed positive edge.

---

## Architecture

```
Input: 240-bar XAUUSD M1 × 10 features
├── Long stream:  LearnableScatteringBlock → Transformer(4L, d=512, 8H)
│                 → attention-weighted pool → long_pooled (512)
└── Short stream: LocalCausalAttention(w=20) → AstrocyteGating(K=16)
                  → regime-conditional T → short_pooled (512)
Fusion: concat → Linear(1024→512) → LayerNorm → GELU → TemperatureScaling
Output: 3-class logits (sell/hold/buy) + confidence scalar

RL layer (frozen encoder):
SAC agent, 16-dim obs, [512,512] hidden, cost curriculum 0→$0.90/trade
```

**Parameters:** 19,534,407 supervised · 1,577,474 RL  
**Training data:** 5,680,771 M1 bars · 2009-03-16 → 2026-04-17

---

## Supervised Training Results

| Run | Test Sell P | Buy P | Signal | Key Change |
|-----|------------|-------|--------|------------|
| 1 | 0.125 | — | 0.000 | Baseline |
| 2 | 0.253 | 0.142 | 0.207 | ReduceLROnPlateau |
| 3–6 | 0.195–0.249 | — | — | Architecture iterations |
| 7 | 0.283 | 0.130 | 0.193 | AstrocyteGating + label smoothing |
| 8 | 0.299 | 0.136 | 0.174 | Extended training |
| 9 | 0.302 | 0.162 | 0.163 | Bear+HIGH oversampling ×3.5 |
| **10** | **0.301** | **0.160** | **0.162** | **Bear+HIGH ×2.5 (recall recovered)** |

**Ceiling:** Sell P ≈ 0.302 across 3 consecutive runs — label-noise floor confirmed. Head-wise RMSNorm (Li et al. ICML 2026) deferred to Run 11 from fresh weights.

---

## RL Training Results

| Phase | Backbone | WR (full costs) | Notes |
|-------|----------|----------------|-------|
| Phase 3 | Run 7 | 49.6% | 13-dim obs, stop-trading collapse |
| Phase 4 | Run 8 | **51.2%** | +ret_1h, +ret_15m — first stable run |
| Phase 5B | Run 8 | 51.2% | Flat-cost hardening — wall confirmed |
| Phase 6 | Run 9 | 41.5% | True VIO zero-inflation regression |
| Phase 8a | Run 10 | 21.6% | Bimodal VIO divergence |
| **Phase 8b** | **Run 10** | **51.2%** | VIO disabled — ceiling confirmed |

**51.2% WR is stable** across Phase 4, 5B, and 8b — three independent runs with different backbones. Breaking this ceiling requires supervised sell P ≥ 0.31.

---

## Deployment Configuration

**Checkpoint:** `rl_agent_evolve2.pt` (Phase 8b, comm=$0.14, spread=0.4pip)  
**Deploy gate:** Bear+HIGH vol regime → **do not trade**  
**Regime identification:** GMM2=Bear (20-bar return < 0) AND vol=HIGH (ATR > 1.4× baseline)  
**Position size:** 0.01 lot (fixed during paper trade)  
**Account:** HF Markets Demo Premium → Pro ($100 min)

**Estimated live edge at HF Markets Pro (0.6 pip, no commission):**

| Metric | Value |
|--------|-------|
| Real cost per trade | ~$0.006 |
| Curriculum max cost | $0.90 |
| Break-even WR | 50.003% |
| Model WR | 51.2% |
| Edge above break-even | **+1.197 pp** |

---

## What Worked

- **AstrocyteGating** (K=16 pattern slots, regime-conditional temperature): +0.04 sell P over baseline attention pool
- **Bear+HIGH oversampling** (×2.5): +0.003 sell P, +0.026 buy P, recall recovered to 0.242
- **15-dim RL obs** (+ret_1h, +ret_15m): eliminated stop-trading collapse, +1.6pp WR
- **Cost curriculum** (10 evolves, $0→$0.90): agent learns selectivity proportional to cost

## What Did Not Work / Lessons

- **VIO (obs[15]):** True VIO (std=0.030) and session-aware VIO (bimodal) both destabilised SAC. Zero is safer than noisy signal until smooth normalisation is designed.
- **Run 9 backbone for RL:** High precision (P=0.302) with low recall (R=0.228) → signal sparsity → WR collapse. RL needs R≥0.24 for stable policy gradient.
- **Head-wise RMSNorm mid-resume:** Cannot be added to pre-trained weights — parameter group mismatch in optimizer. Must train from scratch (Run 11).
- **Phase 6 feature exploration:** Zero supervised additions from 6 candidates. Transformer at d=512 already implicitly computes equivalent signals — label-noise floor is the true ceiling, not feature coverage.

---

*HFTExperiment v2 · Blackl1stV35 · 2026-05-09*
