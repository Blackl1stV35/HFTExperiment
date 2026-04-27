# Phase 4 — Geometric Deep Learning + LLM-Assisted Labelling
## Training Report & Next Phase Proposal

---

## Phase 3 Final Results Summary

| Run | Architecture | Labels | Best test sell P | Best test sell F1 | RL WR | RL best eval |
|-----|-------------|--------|-----------------|------------------|-------|-------------|
| Sup-1 to 3 | InceptionBlock+TCN | Triple-barrier (fixed) | 0.066–0.282 | — | — | — |
| **Sup-5** | **InceptionBlock+TCN** | **Triple-barrier (fixed)** | **0.253** | **0.358** | **44.1%** | **$172** |
| Sup-6 | InceptionBlock+TCN | Triple-barrier (resume) | 0.259 | 0.365 | — | — |
| CCSO-1 | CCSO two-stage (w=20) | Triple-barrier | 0.096 | — | — | — |
| ATR-1 to 3 | InceptionBlock+TCN | ATR-adaptive | ~0.030 val | — | — | — |
| RL-1 to 4 | SAC (ep60 checkpoint) | — | — | — | 44.1% | $172 |

**Definitive conclusion:** Val sell precision is structurally capped at 0.029–0.031 = the dataset base rate (2.94%). The InceptionBlock encoder cannot discriminate M1 OHLCV bars that precede a price decline from those that don't. This is a feature representation problem, not a label quality, architecture depth, or training dynamics problem. Every hyperparameter search has confirmed this ceiling.

---

## Theoretical Foundation: Geometry of Deep Learning (Signal Processing Perspective)

*Reference: Bronstein, Bruna, LeCun, Szlam et al. 2017+; Mallat 2012 (scattering networks)*

### Why the current model fails — a geometric explanation

The InceptionBlock+TCN is equivariant to **time translation** (via causal conv) but not to the symmetries that matter for microstructure:

**Symmetry 1 — Scale equivariance.** A trending market and a ranging market have identical OHLCV bar shapes at different scales. The model should respond identically to a 10-pip move in low-volatility and a 30-pip move in high-volatility — after ATR normalisation. Currently only partially addressed by window_minmax scaling, which does not preserve ATR-relative scale across different regime windows.

**Symmetry 2 — Bid-ask symmetry (price reflection).** Under the transformation `close → -close, buy_label → sell_label`, the prediction should be equivariant. The current model has no mechanism to enforce this. This matters because 89% of training data is in Bull GMM2 state — the model has seen far more buy-setup bars than sell-setup bars, and the representations are asymmetric.

**Symmetry 3 — Microstructure temporal locality.** The signal in M1 data is concentrated in short bursts (spread widening, volume spikes, consecutive directional bars). Mallat's scattering transform is provably stable to diffeomorphisms of the input signal and specifically designed to capture these localised, transient features. The current TCN treats all time steps equally.

---

## Three Geometric Proposals

### Proposal 1 — Ghost Sample Regularisation (Rademacher / PAC-Bayes)

**Theory:** The generalisation gap bound from symmetrisation is:
```
E[L(h)] - Ê[L(h)] ≤ 2 * R_n(H) + O(√(log(1/δ)/n))
```
where `R_n(H)` is the empirical Rademacher complexity. Tightening this bound requires either increasing `n` (impractical) or constraining `H` (the hypothesis class).

**Ghost sample construction for XAUUSD M1:**

For each training sequence `(x_1, ..., x_T, y)`, generate a ghost sample by:

1. **Sign-flip augmentation:** negate all return features (open, high, low, close relative changes), swap sell↔buy labels. This enforces bid-ask symmetry and doubles the effective dataset.

2. **Regime swap:** replace the GMM2 regime embedding with the opposite state (Bear→Bull, Bull→Bear) while keeping the price sequence. The model should produce a calibrated uncertainty increase, not a confident wrong prediction.

3. **Time-scale jitter:** subsample every 2nd bar (simulate M2 perspective), re-label. The model's output on the subsampled sequence should be consistent with the full sequence output — this is a consistency regulariser.

**Implementation:** Add a `GhostSampleAugmenter` to `src/training/labels.py` and call it in the training DataLoader with probability 0.3 per batch. The augmented samples use soft labels (label smoothing 0.1) to prevent the model from memorising the ghost pattern.

```python
class GhostSampleAugmenter:
    """Symmetrisation-based augmentation to tighten Rademacher bound.
    
    Ghost samples enforce:
    1. Bid-ask symmetry: f(-x, swap_label) ≈ 1 - f(x, original_label)
    2. Scale equivariance: f(x/ATR) ≈ f(x/ATR')  for similar regime
    3. Temporal consistency: f(subsample(x)) ≈ f(x) in distribution
    """
    def augment(self, x, y, atr, regime):
        # Sign flip: negate returns, swap sell(0)↔buy(2), hold(1) stays
        x_flip = self._negate_returns(x)
        y_flip = self._swap_labels(y)   # 0↔2, 1→1
        # Soft label with smoothing
        y_soft = F.one_hot(y_flip, 3).float() * 0.9 + 0.033
        return x_flip, y_soft
```

**Expected benefit:** Enforcing bid-ask symmetry directly addresses the Bull-bias problem (89% Bull data). The encoder will learn symmetric representations for sell and buy setups, improving sell precision from the current ~0.03 baseline.

---

### Proposal 2 — Invariant Scattering Features (Mallat)

**Theory:** Mallat's wavelet scattering transform provides:
- Translation invariance up to scale 2^J
- Stability to deformations: `‖Sf(x) - Sf(x∘τ)‖ ≤ C ‖∇τ‖_∞ ‖f‖`
- No information loss (energy-preserving)

**For M1 OHLCV sequences (T=120 bars):** Replace the InceptionBlock's multi-scale convolutions with a learnable scattering network operating on the return series:

```
Layer 0: |x * ψ_{j,r}|  — first-order scattering (ATR-scale wavelets)
Layer 1: ||x * ψ_{j,r}| * ψ_{j',r'}|  — second-order (modulation patterns)
Pool:    S_J[x] = |x * ψ_j| * φ_J  — low-pass envelope
```

The scattering coefficients capture:
- Layer 0: volatility bursts at each M1 scale
- Layer 1: volatility-of-volatility (microstructure regimes within the sequence)
- Pool: the slowly-varying envelope (trend direction)

**Implementation:** Replace `InceptionBlock` with `ScatteringBlock` using `kymatio` (Python scattering library) or a custom learnable wavelet filterbank.

```python
class ScatteringPriceBranch(nn.Module):
    """Price encoder using learnable scattering transform.
    
    Provides geometric stability guarantees not available in
    standard InceptionBlock.
    """
    def __init__(self, T=120, J=4, Q=8, d_model=192):
        # J=4 scales, Q=8 wavelets per octave
        # Output: (T/2^J, C_scatter) scattering coefficients
        ...
```

---

### Proposal 3 — LLM-Assisted Soft Labelling

**Core insight:** The precision ceiling at 0.030 means the hard labels (0/1/2) contain insufficient information to train a discriminative model. A local LLM can generate **soft probability labels** by reasoning about the microstructure context in natural language — information that is present in the feature sequence but not captured by the binary triple-barrier outcome.

**Workflow:**

```
M1 bar sequence → text description → LLM → P(sell), P(hold), P(buy)
```

**Text description template (per sequence):**
```
"XAUUSD M1 sequence, last 10 bars:
 Bar -10: O=2285.3 H=2287.1 L=2284.8 C=2286.2 Vol=847 Spread=0.3
 ...
 Bar -1:  O=2291.4 H=2293.0 L=2290.8 C=2291.9 Vol=1243 Spread=0.6
 ATR(14)=8.2pips. Regime: Bull, vol=HIGH, G/S=Q1.
 Classify directional bias for next 40 bars: sell / hold / buy.
 Return probabilities as JSON: {sell: X, hold: Y, buy: Z}"
```

**LLM options (local, privacy-preserving):**
- `Qwen2.5-7B-Instruct` (4-bit quant, fits T4 GPU alongside training)
- `Mistral-7B-Instruct-v0.3`
- `deepseek-r1:7b` via Ollama

**Integration into training:**

```python
class LLMSoftLabeller:
    """Generate soft probability labels using local LLM.
    
    Used as a pre-training step — run once over training set,
    cache soft labels, use as KL-divergence target alongside hard labels.
    
    L_total = α * L_CE(hard) + (1-α) * L_KL(soft_LLM)
    α anneals from 0.3 (early, trust LLM more) to 0.8 (late, trust model more)
    """
    def generate_soft_labels(self, sequences, prices, regime_info):
        # Batch inference on LLM
        # Returns (N, 3) soft probability array
        ...
```

**Why this directly solves the precision problem:** The hard ATR labels fire a "sell" whenever close drops >0.75×ATR in the next 40 bars, regardless of context. The LLM can reason: "spread is widening, volume spike, consecutive down bars, Bear regime — P(sell)=0.72". This is exactly the kind of context-aware soft signal that the encoder needs to learn discriminative sell representations.

---

## Microstructure Feature Pipeline (Path 2 — Immediate)

Replace the 6-dim OHLCV input with a 10-dim microstructure-aware input in `preprocessing.py`:

```python
# Current: [open, high, low, close, tick_volume, spread]
# Proposed: above + 4 derived microstructure features

derived = [
    # 1. Normalised bar return (direction + magnitude)
    (close - open) / open * 10000,          # in basis points

    # 2. Upper wick pressure (selling pressure above close)
    (high - close) / (high - low + 1e-8),

    # 3. Volume anomaly z-score (rolling 20-bar)
    (tick_volume - roll_mean(tick_volume, 20)) / (roll_std(tick_volume, 20) + 1e-8),

    # 4. Spread pressure (spread relative to bar range — liquidity signal)
    spread / (high - low + 1e-8),
]
```

Update `input_dim: 10` in model config. This is the lowest-risk immediate improvement.

---

## Implementation Priority for Phase 4

| Priority | Proposal | Effort | Expected sell precision gain |
|----------|----------|--------|------------------------------|
| 1 | Microstructure features (10-dim input) | 1 day | 0.03 → 0.06–0.10 |
| 2 | Ghost sample / sign-flip augmentation | 2 days | +0.05–0.10 over base |
| 3 | LLM soft labelling (Qwen2.5-7B local) | 3 days | +0.10–0.20 if labels are calibrated |
| 4 | Scattering transform encoder | 5 days | Uncertain; theoretically grounded |

**Recommended Phase 4 order:** Run P1 first (quick win, confirms feature hypothesis), then P2 alongside P3 (ghost samples + LLM labels can be generated in parallel), then P4 only if P1–P3 plateau below sell P=0.35.

---

## Current Training State

The ep5-saved checkpoint (best_signal_score=0.0281 from ATR run, sell P≈0.030 on val) is live on Drive. Early stopping fired at ep24 with val loss increasing (overfitting). The model has not converged to a useful sell signal.

**Do not proceed to RL with this checkpoint.** The ep60 InceptionBlock checkpoint (sell P=0.253, F1=0.358) remains the best model for RL use.

**Phase 4 starts fresh** with the microstructure feature pipeline as the foundation.

---

## Macro Context (2026-04-27)

| Signal | Value | Implication |
|--------|-------|-------------|
| GMM2 | Bear | Entry gate active in RL |
| Vol | HIGH (19.9%) | Bear+HIGH = Sharpe 0.33 worst cell |
| G/S quartile | Q1 | max_hold=40 bars |
| Deploy recommendation | Reduced size | Wait for vol < 14.3% |
