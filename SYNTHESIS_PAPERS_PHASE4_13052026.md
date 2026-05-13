# SYNTHESIS_PAPERS_PHASE4_13052026.md
**HFTExperiment v2 — Literature Synthesis Phase 4 (Final)**  
**Date:** 2026-05-13  
**Papers reviewed:** 8 + P9 (Alcalde 2025) + P10 (Pion optimizer 2026)  
**Exploration:** `phase4_feature_exploration.ipynb` — all results validated on local NPZ  
**Context:** Phase 8b deployed (51.2% WR, sell P=0.301). Run 11 ready to start.

---

## Final Exploration Results

| Feature | KS | MI | Ratio | Verdict |
|---------|----|----|-------|---------|
| gkv_proxy | 0.080 MOD | 0.000508 (0.05x) | 429.8x bar_return_bps | EXCLUDE |
| d_atr_norm (rolling) | 0.040 WEAK | 0.025 (2.65x) | 1.99x high_sc | EXCLUDE |
| momentum_5bar | 0.196 STRONG | 0.012 (1.29x) | 76.8x bar_return_bps | RL OBS — covered by ret_15m |
| vol_enc_x_ret | 0.242 STRONG | 0.017 (1.79x) | 454.8x bar_return_bps | RL OBS — in encoder probs |
| atr20_norm | 0.764 STRONG | 0.049 (5.30x) | 12.6x spread_pressure | RL OBS — covered by obs[7] |
| **rq_regime** | **0.623 STRONG** | **0.198 (21x)** | **0.60 PASS** | **SUPERVISED** |
| **session_phase_npz** | **0.117 STRONG** | **0.037 (3.99x)** | **2.96 PASS** | **SUPERVISED** |

---

## Key Findings

### session_phase — SUPERVISED (first confirmed addition from all exploration)
KS=0.117 STRONG, MI=0.037 (3.99x OHLCV mean), redundancy ratio=2.96 — passes all three thresholds. Session-phase encodes London/NY trading session presence. London sell rate = 5.07% (1.44x baseline), Asian = 2.69% (0.77x). Feature already exists in NPZ and RL obs (obs[9]) but was never in the 10D supervised feature set. **Patched: added as feature[10] in `preprocessing.py`.** NPZ rebuild required before Run 11.

### DHPF — Strategy reversal confirmed
Bear-SHOCK (vol_enc >= 0.95) has sell% = 14.4% (4x baseline) — already heavily represented, needs NO oversampling. The underrepresented partitions are the Bull partitions where sell labels are scarce despite comprising 87% of data:
- Bull-LOW: sell=1.04% (N=2,068,405, 36.4% of data) → x6.7
- Bull-NORMAL: sell=2.28% (N=2,026,350, 35.7%) → x3.1
- Bear-NORMAL: sell=1.54% (N=13,406, 0.2%) → x4.6

**Patched: `train_supervised.py` DHPF Bull boosts added.** Previous Bear-centric oversampling was correct for sell P improvement but the DHPF result shows the real gap is in Bull-regime sell label exposure.

### Session analysis — London confirmed, VIO gate validated
London (08:00-16:00 UTC) sell rate = 5.07% (1.44x baseline). NY = 0.76x. Asian = 0.77x. VIO session gate (London+NY, 08:00-22:00 UTC) is correct. London is the primary window — aligned with Bangkok time 15:00-23:00 BKK (Thai trader prime hours).

### atr20_norm — strongest KS but redundant with RL obs[7]
KS=0.764 VERY STRONG, MI=0.049 (5.30x) — excellent signal. Ratio=12.6 fails supervised threshold (vs spread_pressure). Already in RL obs as `atr_norm` (obs[7]). No addition needed.

### rq_regime — SUPERVISED confirmed
KS=0.623 STRONG, MI=0.198 (21x OHLCV mean), ratio=0.60 — strongest MI of all features tested across the entire Phase 4-8 exploration. Regime quality scalar from the GMM2 model, range [0, 0.92]. Already in NPZ and RL obs (obs[10]) — zero additional preprocessing cost. **Patched: added as feature[11] in `preprocessing.py`.** Run 11 input is now **12D**.

---

## P10 — Pion Optimizer (Shi et al. 2026, arXiv 2605.12492)
**TIER 2 — DEFERRED to Run 12.**

Updates weight matrices via left/right orthogonal transformations, preserving singular value spectrum throughout training. Competitive with AdamW on LLM pretraining/finetuning. Directly addresses the same spectral drift problem that head-wise RMSNorm fixes post-hoc: Pion prevents eigenspace collapse by construction rather than normalising after the fact.

Run 11 already achieves spectral control via head-wise RMSNorm (Li et al. ICML 2026) + AdamW. Testing Pion would require replacing AdamW (~30 lines in `trainer.__init__`). Deferred to Run 12 if Run 11 still hits the 0.302 sell P ceiling — at that point the optimizer is the remaining untested variable.

---

## Pipeline Patches Applied

| Patch | Status | File |
|-------|--------|------|
| session_phase as 11th supervised feature | APPLIED | `preprocessing.py` |
| feature_dim 10 -> 12 | APPLIED | `dual_branch.yaml` |
| DHPF Bull-LOW/NORMAL/Bear-NORMAL boosts | APPLIED | `train_supervised.py` |
| Bear-SHOCK: NO oversampling | APPLIED | `train_supervised.py` |
| Head-wise RMSNorm | APPLIED (Run 11) | `price_branch_transformer.py` |
| Session-aware VIO (smooth) | APPLIED | `train_rl.py` |
| Directional ATR deploy gate | APPLIED | `paper_trade.py` |
| epochs=200, patience=60 | APPLIED | `config.yaml` |
| Pion optimizer | DEFERRED (Run 12) | — |
| rq_regime as 12th supervised feature | APPLIED | `preprocessing.py` |

---

## Run 11 Procedure

**Step 1 — Rebuild NPZ (required: 11D features now)**
```bash
python scripts/precompute_features.py     --input data/XAUUSD_M1.csv     --output data/training_ready_v2.npz
```
New NPZ will have `features (N, 12)` — session_phase as column[10], rq_regime as column[11].

**Step 2 — Run 11 supervised from scratch**
```bash
python scripts/train_supervised.py     model=dual_branch data=xauusd     training.batch_size=4096     training.epochs=200
```
No `resume_from` — fresh weights required (head-wise RMSNorm + new 11D input layer).  
Target: sell P >= 0.31. Expected: ~20h A100.

**Step 3 — RL Phase 9 (after Run 11, if sell P >= 0.31)**
```bash
python scripts/train_rl.py     --checkpoint "/content/drive/MyDrive/Colab Notebooks/dual_branch_best.pt"     --steps 1500000 --n-evolves 10     --commission 0.70 --spread-pips 2.0     --save-dir "/content/drive/MyDrive/Colab Notebooks/rl_checkpoints_p9"
```
Session-aware VIO (smooth London+NY) re-enabled in `train_rl.py`.  
If WR >= 52%: proceed to HF Markets Pro live ($100 deposit).

---

## Priority Queue (Updated)

| Priority | Action | Status |
|----------|--------|--------|
| NOW | Paper trade Phase 8b evolve2 | Active |
| DONE | rq_regime confirmed SUPERVISED (MI=0.198, 21x OHLCV) | Patched |
| NEXT | Rebuild NPZ with 11D features | Ready |
| NEXT | Run 11 from scratch | Ready after NPZ rebuild |
| AFTER R11 | RL Phase 9 if sell P >= 0.31 | Conditional |
| DEFERRED | Pion optimizer (Run 12) | If R11 ceiling persists |
| DEFERRED | Silver futures XAGUSD feed | Run 12 |

---

*HFTExperiment v2 · SYNTHESIS_PAPERS_PHASE4_13052026.md · 2026-05-13*