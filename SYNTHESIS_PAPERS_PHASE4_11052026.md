# SYNTHESIS_PAPERS_PHASE4_11052026.md
**HFTExperiment v2 — Literature Synthesis Phase 4**  
**Date:** 2026-05-11  
**Papers reviewed:** 8  
**Context:** Evaluated for pipeline addition and architecture patch candidates following Phase 8b deployment (51.2% WR ceiling, supervised sell P=0.301, head-wise RMSNorm deferred to Run 11)

---

## Paper Catalogue

| # | Paper | Method | Relevance tier |
|---|-------|--------|----------------|
| P1 | Ma et al. (2021) — GARCH-MIDAS + Markov regime switching | MS-GARCH-MIDAS, GEPU | **TIER 1** — regime gate improvement |
| P2 | He (2025) — AI gold futures forecast review | SGRU-AM, VMD-ICSS-BiGRU, DBN | **TIER 2** — feature candidates |
| P3 | Farhat & Ghalayini (2020) — Dynamic OLS gold modeling | DOLS, GARCH, Granger | **TIER 3** — macro factor validation |
| P4 | Laduni (2022) — ECM gold futures drivers | ECM, USD index, S&P500, oil | **TIER 3** — macro factor validation |
| P5 | Mensi et al. (2021) — MS-VAR spillovers COVID-19 | MS-VAR, Diebold-Yilmaz | **TIER 1** — regime spillover structure |
| P6 | Goel et al. (2025) — Event-driven gold futures India | Garman-Klass volatility, event study | **TIER 2** — volatility feature candidate |
| P7 | Zhao et al. (2025, Mathematics) — Hybrid LSTM-Transformer-XGBoost | BiLSTM+Transformer, XGBoost feature select | **TIER 1** — architecture alignment |
| P8 | Radev, Golitsis & Mitreva (2023) — Gold ETF COMEX determinants | Newey-West regression, USEPUCIN | **TIER 2** — uncertainty feature candidate |

---

## Tier 1: Directly Actionable

### P1 — Ma et al. (2021): MS-GARCH-MIDAS + GEPU
**Core finding:** GEPU (Global Economic Policy Uncertainty) carries predictive information for gold futures volatility. The Markov regime-switching GARCH-MIDAS model outperforms all competing models. Regime structure matters: the GEPU→volatility relationship is asymmetric — negative GEPU changes (uncertainty rising) are more predictive than positive changes. Two regimes confirmed: low-volatility/normal and high-volatility/crisis.

**Validation against HFTExperiment:**
The existing `cu_au_regime` (obs[10]) and `gmm2` (Bull/Bear) already approximate MS structure. However, P1 identifies that **GEPU-driven volatility regimes are distinguishable from price-trend regimes.** The Bear+HIGH gate currently blocks trading during Bear+HIGH vol — this is partly correct (high vol = GARCH high-volatility regime) but misses the *direction* of uncertainty change. Rising GEPU → higher volatility → existing gate should fire more. Falling GEPU despite high volatility → safe to trade.

**Actionable patch:**
- **Obs feature candidate:** `gepu_delta` — 5-day change in GEPU index, normalised. Can be approximated at M1 from VIX proxy (CBOE VIX futures or GVZ — Gold Volatility Index) as a same-day substitute. MI test required before adding.
- **Deploy gate refinement:** Bear+HIGH gate should also consider *direction* of volatility change. Rising ATR trend → gate fires. Falling ATR trend after spike → gate releases earlier. Implement as `vol_enc = HIGH AND d(ATR)/dt > 0` rather than `vol_enc = HIGH` flat.

---

### P5 — Mensi et al. (2021): MS-VAR Spillovers COVID-19
**Core finding:** Gold is a NET CONTRIBUTOR of spillovers in low-volatility regime but a NET RECEIVER in high-volatility regime (regime reversal). Oil behaves oppositely. Stock markets are primarily self-influenced. The COVID-19 period showed intensified commodity→equity spillovers. US and Chinese gold markets are synchronised with smoothed regime probabilities.

**Validation against HFTExperiment:**
This directly supports the two-regime architecture (GMM2 Bear/Bull). The key new finding is **regime-conditional spillover direction reversal** — gold leads in calm markets but is driven by external shocks in crisis. At M1 resolution, this manifests as:
- Low-vol regime: gold price movement precedes macro announcements (technical structure dominant → model's existing OHLCV features capture this)
- High-vol regime: gold is reactive to cross-market shocks (macro/equity/oil) → OHLCV alone insufficient, external correlators needed

**Actionable patch:**
- **`cu_au_regime` (obs[12])** already partially captures cross-market copper-gold correlation. The P5 finding suggests `us_equity_corr_20bar` (20-bar rolling Pearson between XAUUSD and S&P500 proxy) would discriminate regimes better than a static copper ratio.
- In high-vol regime (vol_enc=HIGH), model should down-weight technical features and up-weight regime/correlation features. The RL agent already gates on regime via obs[10-12] but doesn't have a directional spillover signal.

---

### P7 — Zhao et al. (2025, Mathematics 13:1551): Hybrid LSTM-Transformer-XGBoost
**Core finding:** Three-stage framework: (1) XGBoost feature importance from 36 candidates → 6 key drivers: NASDAQ, S&P500 close, silver futures, USD/CNY, China 1-year Treasury yield, coal ETF; (2) BiLSTM (local temporal) + Transformer cross-attention (global market context); (3) Dynamic Hierarchical Partition Framework (DHPF) stratifying into price trends / volatility / external correlations / event shocks.

**Direct architecture alignment with HFTExperiment:**
The dual-branch design (long stream = Transformer, short stream = LocalCausalAttention) is structurally equivalent to P7's BiLSTM-Transformer interaction. Key differences and gaps:

| Aspect | HFTExperiment current | P7 finding | Gap |
|--------|----------------------|------------|-----|
| Feature selection | Manual 10D OHLCV+micro | XGBoost importance from 36 | Phase 8 validated 10D is MI ceiling — P7 confirms OHLCV insufficient alone |
| External correlators | `cu_au_regime` (static proxy) | Silver futures, USD/CNY, S&P500 | Cross-market signals as direct M1 obs features — not yet tested |
| Temporal coupling | Fixed 240-bar window | Dynamic Hierarchical Partition | DHPF is a training-time data stratification strategy, not inference-time |
| Attention mechanism | Standard causal Transformer | Cross-attention (global-local) | Head-wise RMSNorm (Run 11 pending) is the correct fix; cross-attention is a larger rewrite |
| Event shock handling | None | Explicit event shock partition | Events encoded implicitly via gmm2 regime changes — partial coverage only |

**Actionable patch (Run 11 prep):**
The DHPF concept maps directly onto the existing Bear+HIGH oversampling strategy. Extending to a 4-partition sampler:
1. Price trends (Bull/Bear)
2. Volatility tiers (LOW/NORMAL/HIGH)
3. External correlation regime (cu_au high/low)
4. Event shock (extreme ATR outliers, top 2% of daily ATR)
This is implementable in `build_regime_balanced_sampler()` as a 4-way stratification with separate multipliers.

**The XGBoost feature importance finding** from P7 identifies silver futures (XAG) and USD/CNY as top drivers beyond OHLCV. Both are available via MT5 as additional M1 feeds. Phase 5 should run MI test on these against XAUUSD labels before adding.

---

## Tier 2: Conditionally Actionable (MI Test Required)

### P2 — He (2025): AI Gold Forecast Review
**Key claim:** VMD-ICSS-BiGRU achieved 20.41% annualised return on gold futures trading strategy. SGRU-AM (special GRU + attention) achieves RMSE=4.79, outperforming LSTM. DBN achieves RMSE=0.0557 on normalised price prediction.

**Validation:**
These results are for price-level regression (next-bar close prediction), not 3-class label classification. RMSE=4.79 on gold prices (~$3300 level) = 0.14% per-bar error — competitive but not directly comparable to our precision/recall metrics. The 20.41% annualised return from VMD-ICSS-BiGRU is a long-horizon swing trading result, not M1 scalping.

**VMD (Variational Mode Decomposition)** as a preprocessing step for OHLCV is an interesting candidate. VMD decomposes the price series into K intrinsic mode functions (IMFs) at different frequency bands — equivalent to a learned wavelet decomposition. In the context of HFTExperiment, VMD could replace or augment the LearnableScatteringBlock in the long stream.

**Assessment:** Deferred. VMD preprocessing adds significant complexity and the LearnableScatteringBlock already performs frequency decomposition. Worth benchmarking in Run 12 if RMSNorm (Run 11) doesn't push sell P above 0.32.

---

### P6 — Goel et al. (2025): Event-Driven Gold Volatility India
**Key finding:** Garman-Klass volatility (GKV) captures intraday high-low-open-close volatility more accurately than standard close-to-close. Events confirm: COVID-19, Russia-Ukraine, Israel-Hamas all produced significant abnormal returns (+ve) and elevated GKV. Non-crisis events (elections, G20) had minimal impact.

**Garman-Klass Volatility formula:**
```
GKV[t] = 0.5*(ln(H/L))² - (2*ln2-1)*(ln(C/O))²
```
This uses all four OHLCV prices and is more efficient than `|close-open|/open`.

**Assessment for HFTExperiment:**
`wick_asymmetry` (feature [7]) already partially captures intraday range structure. GKV is a candidate to replace `vol_zscore` (feature [8]) in precompute_features.py — it incorporates H/L gap directly and captures directional bias. MI test required.

**Actionable (Phase 5 candidate):**
```python
# Add to precompute_features.py as gkv_zscore feature:
gkv = 0.5 * np.log(highs/lows)**2 - (2*np.log(2)-1) * np.log(closes/opens)**2
# Replace or augment vol_zscore with rolling z-score of gkv
```

---

### P8 — Radev, Golitsis & Mitreva (2023): Gold ETF COMEX Determinants
**Key finding:** USEPUCIN (US Economic Policy Uncertainty Index) has a strong positive impact on gold ETF prices. Since 2017, CPI and interest rates have *lost* their historical significance — the post-2017 regime is fundamentally different. The 30-year mortgage rate is a surprisingly significant positive predictor.

**Validation against HFTExperiment:**
The post-2017 regime shift is directly relevant — HFTExperiment trains on 2009-2026 data but the 2024-2026 test period sits in the post-2017 regime where traditional macro factors are weaker. This supports the existing decision to use short-term technical/microstructure features rather than macro factors.

**USEPUCIN as a feature:** Daily/monthly frequency — mismatched with M1 bars. Not directly usable as a real-time obs feature without a same-day proxy. The `cu_au_regime` (obs[12]) already serves as a partial uncertainty proxy.

---

## Tier 3: Macro Validation Only

### P3 (Farhat & Ghalayini) and P4 (Laduni)
Both confirm the well-established macro driver hierarchy: USD Index (strongest, negative) > S&P500 (negative, regime-dependent) > Oil (mixed) > Inflation (positive, weakening post-2017) > Fed funds rate (weakest, often insignificant). These validate the exclusion of macro features from the M1 feature set — at M1 resolution, macro moves too slowly to provide label-predictive information (confirmed by Phase 8 MI tests showing all external-factor proxies failed the 0.005 MI threshold).

---

## Architecture Synthesis: What to Add to the Pipeline

### Immediate (before Run 11 training)

**1. DHPF-style 4-partition sampler (train_supervised.py)**

Extend Bear+HIGH oversampling to 4 regime partitions:

```python
# In build_regime_balanced_sampler():
# Partition 4: Extreme event shock bars (ATR > 99th percentile of daily ATR)
# P7 DHPF finding: event shocks need separate treatment from normal HIGH vol
shock_mask  = vol_enc_raw > np.percentile(vol_enc_raw, 99)
regime_mult[bear_mask & shock_mask]                    = 4.0   # Bear+SHOCK
regime_mult[bear_mask & high_mask & ~shock_mask]       = 2.5   # Bear+HIGH (existing)
regime_mult[bear_mask & ~high_mask & ~shock_mask]      = 2.0   # Bear+NORMAL
```

**2. Garman-Klass Volatility as candidate feature (precompute_features.py)**

Replace `vol_zscore` with GKV-based feature. Run MI test in Phase 8 notebook before committing to NPZ rebuild.

```python
# Candidate feature: gkv_norm
gkv_raw = 0.5 * np.log(highs/(lows+1e-8))**2 - \
          (2*np.log(2)-1) * np.log((closes+1e-8)/(opens+1e-8))**2
gkv_norm = np.tanh((gkv_raw - np.median(gkv_raw)) / (np.std(gkv_raw) + 1e-8))
```

**3. Deploy gate: directional ATR for Bear+HIGH (paper_trade.py)**

Replace flat `atr_recent > atr_baseline * 1.4` with directional check (P1 finding):

```python
# In get_regime():
atr_trend = np.mean(atrs[-5:]) - np.mean(atrs[-15:-5])  # rising vs falling
vol_enc   = 1.0 if (atr_recent > atr_baseline * 1.4 and atr_trend > 0) else 0.0
# Only gate when volatility is ACTIVELY RISING, not just elevated
```

---

### Run 11 Architecture Patch (deferred — from scratch)

**Primary (confirmed):** Head-wise RMSNorm in TransformerBlock long stream (Li et al. ICML 2026 — already specced in Phase 3 synthesis).

**Secondary (P7-motivated):** DHPF 4-partition sampler (adds ≤50 lines to `train_supervised.py`).

**Exploratory (Phase 5 investigation):** Silver futures (XAGUSD) and USD/CNY as additional M1 features — requires NPZ rebuild but is the only remaining path to new label-predictive information beyond the current 10D ceiling.

---

## Cross-Paper Consensus: XAUUSD M1 Label-Predictive Signal Hierarchy

Synthesising all 8 papers against HFTExperiment Phase 8 MI results:

| Signal type | Papers supporting | M1 label MI (Phase 8) | Verdict |
|-------------|-------------------|----------------------|---------|
| OHLCV + microstructure | All | 0.008-0.023 | ✓ In use |
| Regime (Bear/Bull, vol tier) | P1, P5, P7 | N/A (structural) | ✓ In use |
| Garman-Klass volatility | P6 | Untested | Test in Phase 5 |
| Silver futures (XAG) | P7 | Untested | Test in Phase 5 |
| USD/CNY exchange rate | P7 | Untested | Test in Phase 5 |
| GEPU / VIX proxy | P1, P8 | ~0.001 est. (daily freq) | Likely fail MI at M1 |
| USD Index | P3, P4, P8 | ~0.001 (daily freq) | Fails at M1 |
| Oil futures | P3, P4, P5 | ~0.001 (daily freq) | Fails at M1 |

**The consistent finding** across all 8 papers is that macro factors (USD Index, Fed rate, CPI, oil) explain gold price *levels* over daily/weekly horizons but have minimal predictive power for intrabar directional signals at M1 — consistent with Phase 8 MI test results.

---

## Summary

| Action | Source | Priority | Effort |
|--------|--------|----------|--------|
| Directional ATR in deploy gate | P1 | NOW (paper_trade.py) | 5 lines |
| GKV feature MI test in .ipynb | P6 | Phase 5 prep | 20 lines |
| DHPF 4-partition sampler | P7 | Run 11 | 30 lines |
| Head-wise RMSNorm | Li et al. ICML 2026 | Run 11 | Confirmed |
| Silver/USD-CNY MI test | P7 | Phase 5 | NPZ test |
| GEPU/VIX proxy | P1 | Deferred | Daily freq mismatch |

---

*HFTExperiment v2 · SYNTHESIS_PAPERS_PHASE4_11052026.md*  
*Next: exploration notebook (phase4_feature_exploration.ipynb) — GKV, silver, USD/CNY MI tests*
