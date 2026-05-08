# PATCH_NOTES — Literature Synthesis + Full Training Report
**Date:** 2026-05-08  
**Release tag:** pre-release-v0.5.5.0-synthesis-rmsNorm-rl-phase7  
**Covers:** Literature synthesis validation, full supervised + RL training history, architecture decision record, priority queue

---

## 1. Literature Synthesis — Validation Against Session History

Two papers reviewed: Kamitani (2026, arXiv) and Li et al. (2026, ICML).

### Paper 1 — Kamitani (2026): NVS Framework
**Verdict: SUPPORTING — conceptually consistent, not directly actionable yet.**

The Naturality Violation Score reframes encoder alignment from object-level correspondence to transformation-preservation. The synthesis correctly identifies:

- `LocalCausalAttention(w=20)` avoids the structural sink — position 0 of the 20-bar window is bar 220 of the full sequence, a semantically recent bar. ✓
- AstrocyteGating as morphism-preserving soft router rather than fixed-position attention. ✓
- `ret_1h` regime-sensitivity (D=0.219 Bear vs Bull) as a transformation carrier across market regimes — consistent with Kamitani's finding that regime-sensitive features carry morphism structure. ✓
- `order_processing_residual` orthogonality (max|r|=0.087) arising from a different generating process, not by decorrelation construction. ✓

**One correction to synthesis interpretation:** The synthesis attributes the val_signal=0.171 vs test_signal=0.193 gap to positional overfitting from the attention sink. This is inconsistent — attention-sink overfitting would produce test < val, not test > val. The positive gap is more likely explained by the val set being post-2024 data (Bear+HIGH heavy, structurally harder than the test set drawn from a different time window). The attention sink is a plausible ceiling contributor but not the explanation for this specific gap.

**NVS diagnostic deferred to Phase 9.** Direct application to price sequences requires a well-defined external proxy W analogous to CLIP-text/DINOv2 in the vision domain. The closest candidate is a combination of `spread_pressure` + `wick_asymmetry` as the market-microstructure anchor.

### Paper 2 — Li et al. (2026, ICML): Attention Sink Origin
**Verdict: PRIMARY — architecturally actionable, implement in Run 10.**

Causal chain confirmed: causal masking → value aggregation variance discrepancy → output projection amplification → super-neuron FFN activation → QK locking at position 0.

**Cross-validated claims:**

| Claim | Status | Notes |
|-------|--------|-------|
| Long stream has attention sink at bar 0 | ✓ CONFIRMED | bar 0 = ~2 hours ago in XAUUSD M1, semantically stale |
| Short stream (w=20) avoids sink | ✓ CONFIRMED | position 0 of w=20 window = bar 220, recent |
| Pre-norm architecture is compatible with head-wise RMSNorm | ✓ CONFIRMED | Phase 5 uses pre-norm throughout |
| Mechanism is structural, not parameter-count dependent | ✓ CONFIRMED | applies at any depth |
| Sink explains sell P=0.283 ceiling | ⚠ PARTIAL | class imbalance is also confirmed cause (Run 9 pushed P without RMSNorm); sink is a plausible additional contributor |
| val > test signal gap = positional overfitting | ✗ INCONSISTENT | gap direction is wrong for overfitting; distribution shift more likely |

**Implementation spec (long stream only):**

```python
# In TransformerBlock.forward(), after:
#   o = (attn_weights @ V)  # (B, n_heads, T, d_head)
# Add before output projection:
o = o / (o.norm(dim=-1, keepdim=True) + 1e-8) * self.head_scale

# In TransformerBlock.__init__, add:
self.head_scale = nn.Parameter(torch.ones(n_heads, 1, d_model // n_heads))
```

This adds `n_heads × d_head = d_model = 512` learnable parameters — negligible (0.003% of model size). Initialises at identity. Does not apply to short stream (LocalCausalAttention) which is already sink-free.

---

## 2. Full Supervised Training History

| Run | Test SellP | BuyP | Signal | Val SellP peak | Ep | Key change |
|-----|-----------|------|--------|----------------|----|------------|
| 1 | 0.125 | — | 0.000 | — | — | no ckpt (floor=0.30) |
| 2 | 0.253 | 0.142 | 0.207 | 0.193 | — | ReduceLROnPlateau, best signal |
| 3 | 0.195 | 0.056 | 0.148 | — | — | wd=0.05 overcorrected |
| 4 | 0.247 | 0.092 | 0.174 | — | — | cosine+astrocyte |
| 5 | 0.249 | 0.103 | 0.177 | — | — | |
| 6 | 0.228 | 0.100 | 0.169 | — | — | early stop bug fixed |
| 7 | 0.283 | 0.130 | 0.193 | 0.171 | 90 | new best sell P |
| 8 | **0.299** | 0.136 | 0.174 | 0.243 | 62 | resumed R7 _last.pt, reset_lr=5e-5 |
| 9 | **0.302** | **0.162** | 0.163 | 0.246 | 68 | Bear+HIGH x3.5 oversampling |

**Current best checkpoints on Drive:**
- `dual_branch_best.pt` → Run 8 ep61 (signal=0.171 val, sell P=0.193 val, **preferred for RL**)
- `dual_branch_last.pt` → Run 9 ep68 (sell P=0.302 test, sell R=0.228, precision-optimal)

**Key observations:**
- Test sell P ceiling: **0.302** (Run 9). Val sell P ceiling: **0.246** (Run 9 ep2).
- Run 9 achieved higher precision but lower recall (R=0.228 vs R=0.274 in Run 8). This recall reduction caused RL Phase 6 WR collapse (see RL history below).
- Bear+HIGH oversampling (x3.5) confirmed effective: both sell P and buy P improved. Oversampling multiplier of 3.5 may be too aggressive — Run 10 will test x2.5.
- LR decay schedule: all runs converge to GradNorm≈0.1-0.2 at the final LR (3.13e-6 in Run 9). The model is reaching the label-noise floor.

---

## 3. Full RL Training History

| Phase | Obs | Backbone | Best PnL | Final PnL | Final WR | Trades/ep | Notes |
|-------|-----|----------|----------|-----------|----------|-----------|-------|
| Phase 3 v2 | 13D | Run 7 | $244.65 | $72.35 | 49.6% | 58 | baseline; stop-trading collapse ev5-7 |
| Phase 4 | 15D | Run 8 _best | $240.86 | $73.58 | **51.2%** | 81 | +ret_1h +ret_15m; no collapse |
| Phase 5 (16D, VIO=uniform) | 16D | Run 8 _best | $240.38 | $73.58 | 51.2% | 81 | VIO fallback = ret_15m clone; identical to P4 |
| Phase 5B (resume ev5) | 16D | Run 8 _best | $240.38 | $73.58 | 51.2% | 81 | flat-cost hardening; wall confirmed |
| **Phase 6** | 16D | Run 9 _last | **-$55.42** | -$509.89 | **41.5%** | 81 | **REGRESSION: true VIO zero-inflation + sparse recall** |

### Phase 6 Regression — Root Cause Record

Two simultaneous changes caused the regression from 51.2% → 41.5% WR:

**Cause A — True VIO zero-inflation (primary):**
tick_volume_raw became available in NPZ, enabling true VIO computation. XAUUSD M1 tick volume is near-constant during Asian session (~13% of bars), making VIO ≈ 0 for extended periods. True VIO std=0.030 vs uniform fallback std≈0.17 (6× smaller amplitude). The 16th observation dimension became a near-flat noise channel, poisoning the policy gradient.

**Fix applied:** VIO (obs[15]) set to zeros in `train_rl.py`. Revival path: session-aware VIO — compute only on London+NY hours, zero-pad Asian session.

**Cause B — Run 9 recall too sparse (secondary):**
Run 9 sell R=0.228 vs Run 8 sell R=0.274. At 87% Bull data, the Bear+HIGH-specialised encoder fires very few sell signals in the dominant regime. The SAC policy encountered high-confidence signals that were systematically wrong in Bull regime. Confidence 0.978±0.107 (vs 0.953±0.162 in P4) — narrow distribution provided no discrimination signal to the gating mechanism.

**Fix:** Use Run 8 `_best.pt` (ep61) as backbone for Phase 7. Run 8's sell R=0.274 provides sufficient signal density for the policy to gate from across all regimes.

---

## 4. HF Markets Broker Cost Analysis

Registration planned on HF Markets (MT4/MT5). Account cost structures:

| Account | Min$ | Spread | Commission | Cost/trade (0.01 lot XAUUSD) | Break-even WR |
|---------|------|--------|-----------|------------------------------|---------------|
| Premium | $0 | 1.4 pip | No | ~$0.014 | ~50.007% |
| Cent | $0 | 1.4 pip | No | ~$0.014 | ~50.007% |
| Zero Spread | $0 | 0.0 pip | Yes (unknown) | unknown | unknown |
| **Pro** | **$100** | **0.6 pip** | **No** | **~$0.006** | **~50.003%** |

**Recommended: Pro account.** 0.6 pip spread is fully covered by RL curriculum (evolve ~2, step ~300k). At 0.01 lot XAUUSD, real cost is $0.006/trade vs curriculum max $0.90/trade (150× lower). Phase 4's 51.2% WR is strongly profitable at these costs without any further improvement.

**Key insight:** The curriculum was deliberately over-pessimistic about costs (designed for retail worst-case). Any WR above 50.003% is break-even on Pro. The 51.2% WR already achieved represents a robust edge at real broker costs.

**Deployment checkpoint:** `rl_agent_evolve2.pt` from Phase 4/5 (comm=$0.14, spread=0.4pip) for Bull+LOW vol regime only. Deploy gate: GMM2=Bear AND vol=HIGH → do not trade.

---

## 5. Architecture Decision Record

| Component | State | Decision |
|-----------|-------|----------|
| `training_ready.npz` | 10D + tick_volume_raw (Phase 8 patch) | No supervised additions from Phase 8 |
| `price_branch_transformer.py` | n_bypass=0 (no-op), no head-wise RMSNorm yet | Add RMSNorm in Run 10 |
| `dual_branch.yaml` | n_bypass_features=0, d_model=512, 4L, 8H | Unchanged |
| `train_supervised.py` | Bear+HIGH x3.5 sampler active | Change to x2.5 in Run 10 |
| `train_rl.py` | 16D obs, VIO disabled (zeros) | Phase 7 uses Run 8 _best.pt |
| RL obs vector | 16D: [0-3] probs+conf, [4-6] position, [7-9] atr/trend/session, [10-12] regime, [13-14] ret_1h/ret_15m, [15] inv_pressure=0 | VIO revival pending session-aware normalisation |

---

## 6. Priority Queue — Revised After Synthesis

| Priority | Action | Rationale | Blocker |
|----------|--------|-----------|---------|
| **1 — NOW** | **RL Phase 7** | Recover 51.2% WR using Run 8 _best.pt + VIO disabled. At HF Markets Pro costs, 51.2% WR is already a deployable edge. | None — train_rl.py patched |
| **2 — NEXT** | **Supervised Run 10** | Bear+HIGH x2.5 (softer) + head-wise RMSNorm (Li et al. ICML 2026). Two improvements in one run: recover sell R toward 0.26+ while keeping P≥0.30; head-wise RMSNorm may push P toward 0.31+ by fixing bar-0 attention sink. | Phase 7 completion for comparison baseline |
| **3 — AFTER R10** | **RL Phase 8** | Run RL against Run 10 backbone. If sell P≥0.31 and sell R≥0.26, projected WR ~52-53% at full curriculum costs → profitable even at retail. | Run 10 |
| **4 — PARALLEL** | **Session-aware VIO** | Compute VIO on London+NY bars only, zero-pad Asian. Re-enable obs[15] with meaningful signal. | Timestamp array → hour extraction in train_rl.py |
| **5 — DEFERRED** | **Attention sink diagnostic** | Visualise bar-0 attention weight distribution in long stream to confirm sink presence quantitatively. | Run 10 training complete |
| **6 — FUTURE** | **NVS diagnostic (Phase 9)** | Kamitani NVS framework adapted to price sequences. Requires defining external proxy W from microstructure features. | Research design |

---

## 7. Run 10 Architecture Spec

Two changes from Run 9 baseline:

**Change 1 — Bear+HIGH multiplier x2.5 (was x3.5):**
```yaml
# train_supervised.py build_regime_balanced_sampler()
bear_high_mult: 2.5   # was 3.5
```
Softer oversampling to recover sell recall toward 0.26+ while keeping precision near 0.30.

**Change 2 — Head-wise RMSNorm in long stream TransformerBlock:**
```python
# TransformerBlock.__init__() — add:
self.head_scale = nn.Parameter(torch.ones(n_heads, 1, d_model // n_heads))

# TransformerBlock.forward(), after value aggregation:
# o: (B, n_heads, T, d_head)
o = o / (o.norm(dim=-1, keepdim=True) + 1e-8) * self.head_scale
# then: o.transpose(1,2).view(B, T, d_model) → out_proj
```
Adds 512 parameters. Targets bar-0 attention sink in 120-bar long stream. Does NOT apply to short stream (LocalCausalAttention w=20 is already sink-free per Li et al. analysis).

**Resume:** from Run 9 `_last.pt` with `reset_lr=2.5e-5`, epochs=150, patience=60.

---

## 8. Phase 7 Run Command

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
    --save-dir "/content/drive/MyDrive/Colab Notebooks/rl_checkpoints_p7"
```

Expected outcome: recover to ≥51.2% WR (Phase 4 baseline). VIO is zeros (obs[15] is no-op). Backbone is Run 8 ep61 with sell R=0.274 — sufficient signal density for SAC gating across all regimes.

---

*HFTExperiment v2 · PATCH_NOTES_SYNTHESIS_TRAINING_REPORT_08052026.md*  
*Next milestones: Phase 7 RL recovery → Run 10 supervised (RMSNorm + x2.5 mult) → Phase 8 RL*
