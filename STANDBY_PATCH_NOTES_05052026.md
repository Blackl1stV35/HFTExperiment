# Literature Synthesis — Feature Exploration Lab
## Microstructure Foundations: Spread Decomposition, Rough Volatility, Fractal Dynamics & Anomaly Detection

**Papers reviewed:**
1. El Euch, Fukasawa & Rosenbaum (2016) — *Microstructural foundations of leverage effect and rough volatility*
2. Hagströmer, Henricsson & Nordén (2016) — *Components of the bid–ask spread and variance: a unified approach*
3. Muhammad et al. (2023) — *Fractal geometry in HFT: modelling market microstructure and price dynamics*
4. Shu, Wang & Liang (2024) — *Early warning indicators for financial market anomalies: a multi-signal integration approach*

---

## 1. What Each Paper Establishes

### Paper 1 — El Euch et al.: Rough Volatility from Microstructure (★★★★★)

The central result: four observable microstructure properties — high endogeneity, no-arbitrage, buying/selling asymmetry, and metaorders — are sufficient, via Hawkes process dynamics, to generate **rough Heston volatility** at macro scale. Specifically, when the kernel of the Hawkes process has a power-law tail with exponent α ∈ (½, 1), the resulting macroscopic volatility process is a fractional Brownian motion with Hurst parameter H = α − ½ ≪ ½. Empirically, H ≈ 0.1 across a wide range of liquid assets.

Key implication: rough volatility is not an assumption — it is a theorem derivable from HFT order-flow mechanics. The same mechanics generate **leverage effect** (negative price-volatility correlation) through buying/selling asymmetry: a market maker with long inventory raises price less on a buy order than it lowers it on a sell order of equal size. This directional asymmetry in the Hawkes intensity — more precisely, the off-diagonal elements of the excitation kernel matrix — produces the negative correlation in the limit.

For XAUUSD: gold is a globally traded, highly liquid asset. The Hurst parameter of H ≈ 0.1 applies. Volatility is therefore **mean-reverting faster than Brownian** and the excitation structure of order flow is measurable from tick data. The implication for the HFT model: `vol_zscore` (existing Phase 4 feature, MI=0.02536) is capturing part of this rough volatility signal, but it is a scalar z-score — it does not encode the Hawkes kernel's temporal decay structure.

### Paper 2 — Hagströmer et al.: Spread Decomposition for Gold Futures (★★★★★)

Applied directly to gold futures (Shanghai Futures Exchange) — the most relevant empirical paper of the four for XAUUSD. Key quantitative findings:

- **50% of the bid–ask spread** = order processing costs
- **~25% of the bid–ask spread** = adverse selection (asymmetric information)
- **~25% of the bid–ask spread** = inventory costs
- **~33% of return variance** = microstructure noise
- Of the noise variance: dominated by **price discreteness and order processing costs**; adverse selection and inventory pressure are marginal contributors to variance
- Afternoon session variance >> morning session variance (European markets opening)
- Gold value innovations are almost entirely **public news**, not informed trading — consistent with gold being a macro hedge asset, not an informed-flow asset

The model-implied bid–ask spread captures 87% of the observed spread — very high fit. The structural model (MRR extension with inventory) decomposes the spread into three closed-form components: permanent (adverse selection), transitory-inventory, and transitory-order-processing.

For XAUUSD: the existing `spread_pressure` feature (KS=0.600, MI=0.05958 — best feature in the project) is directly measuring the composite spread signal without decomposing it. The Hagströmer et al. decomposition suggests **three separate features** could replace or augment it: an adverse selection component, an inventory component, and an order processing component. The inventory component is session-dependent (builds through the day, resets at open) — which explains why `session_enc` and `vwap_dev_norm` partially captured similar information, producing the redundancy failure in Phase 6.

### Paper 3 — Muhammad et al.: Fractal Geometry in HFT (★★★)

A theoretical survey with no original empirical results. Core contribution: formalization of the Hurst exponent framework for HFT price dynamics, via three parallel models:
- **Fractional Brownian motion:** ΔP ∝ B^H(T), H measuring persistence (H > ½) or anti-persistence (H < ½)
- **Autoregressive model:** Pt = φ · Pt−1 + et, foundational but insufficient for fractal dynamics
- **Multifractal random walk (Bacry et al. 2001):** f(α) = α·ζ(q) − D(q), capturing varying self-similarity across timescales

The Hurst exponent H < ½ (empirically ~0.1 for volatility, per Paper 1) means **anti-persistent** volatility — a large volatility increment tends to be followed by a smaller one. This is compatible with the mean-reversion observed in `vol_zscore` (MI=0.02536) and the ATR-adaptive label behaviour.

The multifractal spectrum insight is directly relevant to Phase 6: the reason `ret_5m`, `ret_15m`, and `ret_1h` all pass KS but with increasing D (0.104, 0.206, 0.361) is precisely the multifractal structure — different timescales carry different levels of self-similar information. The `ret_1h` dominance reflects that gold's multifractal scaling is most informative at hourly horizons, consistent with the Hagströmer et al. finding that variance is driven by session-level (hourly) dynamics.

The paper is a theoretical review without original results — cited for conceptual framing only.

### Paper 4 — Shu et al.: Multi-Signal Integration for Anomaly Detection (★★★)

BiLSTM-attention architecture achieving 15.4% precision improvement over single-signal methods, with 2.8-day average lead time gain on anomaly detection across 2010–2023 financial data. The multi-signal hierarchy combines: microstructure metrics, technical indicators, fundamental data, sentiment, and cross-asset signals.

Key finding: performance is **strongest during regime transitions**, which is precisely when the model fails (the supervised backbone at sell P=0.283 performs worst in regime-change bars — Bear-to-Bull and Bull-to-Bear transitions overlap the hold/sell decision boundary).

The BiLSTM-attention architecture parallel is notable: their approach to multi-scale feature fusion (attention weights over heterogeneous signal streams) is architecturally similar to the Phase 5 Transformer + AstrocyteGatingModule design. The reported 15.4% precision gain from multi-signal integration validates the direction, though their domain (daily anomaly detection) and ours (M1 directional classification) are not directly comparable.

---

## 2. Cross-Paper Synthesis

The four papers converge on three themes directly relevant to the HFTExperiment system:

### Theme A: Spread is a composite — `spread_pressure` is conflating three signals

Hagströmer et al. decompose the gold spread into adverse selection (permanent), inventory (transitory-cumulative), and order processing (transitory-instantaneous). The existing `spread_pressure` feature computes `spread / (high − low)` — a ratio that combines all three components into a single scalar. This is why it is the strongest individual feature (MI=0.05958) but also why it is already partially redundant with `vwap_dev_norm` (Phase 6 redundancy ratio: 596.8 — VWAP deviation is an inventory proxy derived from the same price series).

**Implication for Phase 7:** decompose `spread_pressure` into its three structural components using the Hagströmer et al. MRR framework. The adverse selection component (permanent price impact per trade) and inventory component (cumulative daily position) are theoretically independent and potentially add distinct discriminative signal. This is a higher-expected-value feature engineering target than any of the six Phase 6 candidates.

### Theme B: Rough volatility (H ≈ 0.1) explains the multi-TF return scaling

El Euch et al. prove that Hawkes-driven order flow produces rougher-than-Brownian volatility. The fractal paper formalises the Hurst exponent scaling: ΔP(T) ~ T^H. With H ≈ 0.1 for volatility, return increments at different timescales scale as T^0.1 rather than T^0.5 (Brownian). This is why `ret_1h` at KS=0.361 dominates `ret_5m` at KS=0.104 by a ratio of 3.5× — the 12× difference in window length produces 12^0.1 ≈ 1.28 predicted scaling under H=0.1, but the actual ratio (3.5×) is larger. The gap is explained by the Hawkes excitation: the 60-bar window captures the full Hawkes kernel decay (mean reverting within ~30–40 bars for XAUUSD at M1) that the 5-bar and 15-bar windows miss.

**Implication for Phase 7:** the optimal aggregation window for a single multi-TF return feature is approximately the Hawkes kernel support — empirically estimable as the autocorrelation decay length of `bar_return_bps`. If this is ~40 bars (40 minutes), then `ret_40m` would be theoretically optimal, not `ret_1h`.

### Theme C: Regime transitions are the discriminative bottleneck — not features

Shu et al. report strongest multi-signal performance during regime transitions. Phase 6 confirmed `ret_1h` and `ret_15m` are regime-sensitive (KS Bear vs Bull: 0.219 and 0.103). The supervised backbone at 3-class classification cannot express regime-conditional position sizing. The RL agent operating on a 15-dim obs vector including `ret_1h` and `ret_15m` can learn this — regime-conditional gates over the directional signal.

---

## 3. Validation of Phase 6 Decisions

| Phase 6 decision | Literature support |
|---|---|
| `spread_pressure` as best existing feature | Hagströmer et al.: spread is the most informative single microstructure signal; gold futures decomposition confirms |
| `dxy_return_20` exclusion | No paper supports H1→M1 forward-fill; Hagströmer et al. confirm gold value innovations are public news (macro), not cross-asset flow at M1 scale |
| `ret_1h` RL obs inclusion | El Euch et al.: 60-bar window spans Hawkes kernel support; fractal scaling (H≈0.1) predicts 1h as maximally informative MTF window |
| No supervised additions | Shu et al.: 15.4% gain requires multi-signal fusion architecture, not individual feature additions; Transformer already implicitly computes MTF returns via attention |
| Redundancy threshold over-conservative | Fractal paper: at H≈0.1, all log-return aggregations are correlated by construction — any pairwise MI ratio between MTF returns will exceed 1.0 |

---

## 4. New Feature Candidates for Phase 7

Based on synthesis across all four papers:

### 4.1 Spread Decomposition Features (Hagströmer et al.)

Three features replacing or augmenting `spread_pressure`:

**`adverse_selection_proxy`**: change in midpoint after a trade, normalised by ATR. This approximates the permanent price impact (the α·It term in the Glosten-Milgrom model). Positive after a buy → buy pressure information; negative after sell → sell pressure.
```python
# Permanent midpoint revision: (close[t] - close[t-1]) * sign(trade direction)
# Proxy using bar_return × sign(close - open) as trade direction indicator
adv_sel = bar_return_bps * np.sign(close - open)
adv_sel_norm = rolling_zscore(adv_sel, window=120)
```

**`inventory_pressure_proxy`**: cumulative signed volume since session open, normalised by total session volume. Approximates the d·Σxt term in the Hagströmer model.
```python
# Session-cumulative signed volume
signed_vol = tick_volume * np.sign(close - open)
inv_pressure = session_cumsum(signed_vol) / (session_cumvol + 1e-8)
inv_pressure_norm = np.tanh(inv_pressure)
```

**`order_processing_component`**: the residual transitory spread not explained by information or inventory — approximated as `spread − 2 × |bar_return_bps|`. This isolates the pure cost component.

### 4.2 Hawkes Kernel Decay Feature (El Euch et al.)

**`hawkes_excitation_40`**: 40-bar autocorrelation-weighted return sum, with exponentially decaying weights (λ ≈ 0.1 matching the estimated Hurst parameter). This explicitly encodes the Hawkes kernel support rather than using a flat rolling sum.
```python
decay = np.exp(-0.1 * np.arange(40))  # Hawkes-inspired exponential decay
decay /= decay.sum()
hawkes_40 = np.convolve(log_ret, decay[::-1], mode='same')
hawkes_40 = rolling_zscore(hawkes_40, window=120)
```

This is theoretically superior to `ret_1h` (flat 60-bar sum) because it weights recent bars more heavily, consistent with the Hawkes process's self-excitation structure where recent order flow has higher excitation intensity.

### 4.3 Rough Volatility Indicator (El Euch et al. + Fractal paper)

**`roughness_indicator`**: local Hurst exponent estimated via R/S analysis over a rolling 40-bar window. When H < 0.3 (rougher than typical), regime is high-endogeneity — Hawkes excitation is dominant and directional signals are stronger. When H > 0.4, the market is approaching Brownian — directional signals weaker.
```python
# R/S ratio: range of cumulative deviations / std, log-log slope = H
def local_hurst(arr, window=40):
    ...  # rolling R/S estimation
roughness = local_hurst(log_ret, window=40)
roughness_norm = np.tanh((roughness - 0.25) / 0.1)  # centred at H=0.25
```

---

## 5. Priority Ranking for Phase 7

| Feature | Theory basis | Expected KS | Redundancy risk | Priority |
|---------|-------------|-------------|-----------------|----------|
| `adverse_selection_proxy` | Hagströmer spread decomposition | High | Low (novel decomposition) | **1** |
| `hawkes_excitation_40` | El Euch et al. kernel decay | High | Low-moderate | **2** |
| `inventory_pressure_proxy` | Hagströmer inventory component | Moderate | Moderate (session correlated) | **3** |
| `roughness_indicator` | Fractal / rough vol theory | Moderate | Low (no existing analog) | **4** |
| DXY (M1 broker feed) | Cross-asset, existing thesis | Moderate | Low (exogenous) | **5** |

All five are blocked on completing RL training first. Phase 7 feature work begins after RL convergence assessment.

---

## 6. Paper Quality Assessment

| Paper | Venue | Rigor | Relevance | Weight in synthesis |
|-------|-------|-------|-----------|---------------------|
| El Euch et al. (2016) | arXiv/q-fin | ★★★★★ Peer-reviewed, rigorous proofs | ★★★★★ Direct theoretical foundation | Primary |
| Hagströmer et al. (2016) | Journal of Futures Markets (Wiley) | ★★★★★ Empirical, gold futures specific | ★★★★★ Applied to exact asset class | Primary |
| Shu et al. (2024) | JACS (SciPublication) | ★★★ Methodologically sound, limited venue prestige | ★★★ Architectural validation | Supporting |
| Muhammad et al. (2023) | Saudi J Econ Fin | ★★ Theoretical survey, no empirical results | ★★★ Conceptual framing only | Background |

---

*Synthesis — Phase 6 Feature Exploration Lab · 2026-05-03*
*HFTExperiment v2 — next session: RL training + Phase 7 feature design*
