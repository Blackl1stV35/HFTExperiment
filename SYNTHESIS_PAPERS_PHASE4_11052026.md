# SYNTHESIS_PAPERS_PHASE4_11052026.md
**HFTExperiment v2 — Literature Synthesis Phase 4**  
**Date:** 2026-05-11 | **Updated:** 2026-05-13  
**Papers reviewed:** 8 + 1 (Alcalde et al. 2025 concentration phenomena)  
**Exploration:** `phase4_feature_exploration.ipynb` — validated on local NPZ with internal reconstruction  
**Context:** Phase 8b deployed (51.2% WR, sell P=0.301, head-wise RMSNorm deferred to Run 11)

---

## Paper Catalogue

| # | Paper | Tier |
|---|-------|------|
| P1 | Ma et al. 2021 — MS-GARCH-MIDAS + GEPU | **TIER 1** — regime gate |
| P2 | He 2025 — AI gold futures review | TIER 2 |
| P3 | Farhat & Ghalayini 2020 — DOLS gold modeling | TIER 3 |
| P4 | Laduni 2022 — ECM gold futures | TIER 3 |
| P5 | Mensi et al. 2021 — MS-VAR COVID spillovers | **TIER 1** — regime structure |
| P6 | Goel et al. 2025 — Event-driven gold volatility | TIER 2 |
| P7 | Zhao et al. 2025 — Hybrid LSTM-Transformer-XGBoost | **TIER 1** — architecture |
| P8 | Radev et al. 2023 — COMEX determinants | TIER 2 |
| P9 | Alcalde et al. 2025 — Mean-field Transformer concentration | **TIER 1** — theoretical |

---

## Exploration Results (Validated)

Local NPZ keys confirmed: `features(N,10)`, `labels`, `close`, `high`, `low`, `timestamps_ns`, `gmm2`, `vol_enc`, `tick_volume_raw`, `rq`, `atr_norm`, `trend_norm`, `session_phase`.  
Note: `features[:,8]` (vol_zscore) and `features[:,9]` (spread_pressure) are broken in local NPZ — reconstruction via rolling `tick_volume_raw` zscore also failed (sparse vol_sc). Results remain valid because labels and the 8 intact features are correct.

| Feature | KS | MI | Ratio | Verdict |
|---------|----|----|-------|---------|
| gkv_proxy | 0.0795 MOD | 0.000508 (0.05x OHLCV) | 429.8x bar_return_bps | **EXCLUDE** |
| d_atr_norm (rolling) | 0.0398 WEAK | 0.024679 (2.65x OHLCV) | 1.99x high_sc | **EXCLUDE** |
| momentum_5bar | 0.1956 STRONG | 0.011043 (1.19x OHLCV) | 72.6x bar_return_bps | **RL OBS** |
| DHPF partitions | — | — | needs vol_enc | **PENDING** |

---

## Feature Verdicts

### GKV — EXCLUDE (confirmed)
MI=0.0005 (5% of OHLCV mean). Ratio=430. Consistent across two separate test runs. MinMax-scaled OHLC makes H-L range near-constant so GKV degenerates to near-zero variance. Not actionable. `vol_zscore` retained.

### d_ATR (rolling) — EXCLUDE
True rolling 5-bar vs 15-bar ATR gives KS=0.040 WEAK. The previous KS=0.149 STRONG result was an artifact of the broken `vol_zscore` column being used as a proxy. Real rolling d_ATR has no label distribution separation.

The **deploy gate patch in `paper_trade.py`** remains valid — it uses live bar ranges at runtime, not label-predictive signal. No supervised or RL addition.

### momentum_5bar — RL OBS (verdict, but no action)
KS=0.196 STRONG, MI=0.011. Ratio=72.6 fails supervised threshold. For RL obs: `ret_15m` (obs[14]) already covers 15-bar z-scored return — 5-bar momentum is redundant with a shorter window. **No addition needed.**

### DHPF Partitions — PENDING
All runs failed because `vol_zscore` reconstruction produced all-zeros (same root cause as col[8]). `vol_enc` from the NPZ is the correct column. Replace Cell 7 with:

```python
vol_enc_raw = _d["vol_enc"].astype(np.float64)[:N]
gmm2_raw    = _d["gmm2"].astype(np.float64)[:N]
bull   = gmm2_raw > 0.5
low_v  = vol_enc_raw < 0.25
norm_v = (vol_enc_raw >= 0.25) & (vol_enc_raw < 0.75)
high_v = (vol_enc_raw >= 0.75) & (vol_enc_raw < 0.95)
shock_v = vol_enc_raw >= 0.95
parts = {
    "Bull-LOW":    bull  & low_v,  "Bull-NORMAL": bull  & norm_v,
    "Bull-HIGH":   bull  & high_v, "Bull-SHOCK":  bull  & shock_v,
    "Bear-LOW":   ~bull  & low_v,  "Bear-NORMAL":~bull  & norm_v,
    "Bear-HIGH":  ~bull  & high_v, "Bear-SHOCK": ~bull  & shock_v,
}
```

---

## P9 — Alcalde et al. 2025: Transformer Concentration Phenomena

**SUPPORTING — theoretical grounding for Run 11 RMSNorm.**

Proves deep encoder Transformers concentrate onto dominant eigenspace of `V*B^T` within O(log beta) steps at inference. Bar-0 attention sink is a direct instance: causal masking makes bar 0 the low-energy attractor of the Lyapunov functional. Li et al. ICML 2026 give the practical fix (head-wise RMSNorm breaks `V*B^T` locking); Alcalde et al. provide the theoretical proof. Both papers are consistent and mutually reinforcing.

**Run 11 diagnostic:** after training, visualise attention weight distribution at bars 0, 60, 119 to confirm RMSNorm suppressed the collapse. If bar-0 weights still dominant, increase head_scale learning rate.

---

## Pipeline Patches Summary

| Patch | Status | File |
|-------|--------|------|
| Directional ATR deploy gate | APPLIED | `paper_trade.py` |
| GKV supervised addition | EXCLUDED | — |
| d_ATR supervised/RL addition | EXCLUDED (KS fails) | — |
| momentum_5bar RL obs | EXCLUDED (ret_15m covers) | — |
| DHPF Bear-SHOCK sampler | PENDING retest with vol_enc | `train_supervised.py` |
| Head-wise RMSNorm | Run 11 (fresh weights) | `price_branch_transformer.py` |

---

## Priority Queue

| Priority | Action | Status |
|----------|--------|--------|
| NOW | Paper trade Phase 8b evolve2 on HF Markets Demo Premium | Active |
| NEXT | DHPF corrected retest using vol_enc from NPZ | Ready |
| NEXT | Run 11 supervised from scratch + head-wise RMSNorm | Ready |
| AFTER R11 | RL Phase 9 if sell P >= 0.31 | Conditional |
| DEFERRED | Silver futures (XAGUSD) MI test | Run 12 |
| DEFERRED | Smooth VIO normalisation | Research |

---

*HFTExperiment v2 · SYNTHESIS_PAPERS_PHASE4_11052026.md · Updated 2026-05-13*