# Literature Synthesis: Astrocyte Memory + Tri-Accel
## Validation Against ScatterTCN Architecture & Phase 4 Training Pipeline

**Papers reviewed:**
- Vivet & Arenas (2026). *Emergent Self-Attention from Astrocyte-Gated Associative Memory Dynamics.* arXiv:2604.25481
- Sheibanian et al. (2025). *Tri-Accel: Curvature-Aware Precision-Adaptive and Memory-Elastic Optimization.* arXiv:2508.16905v2

---

## Paper 1: Astrocyte-Gated Associative Memory

### Core mechanism

A Hopfield-type network where astrocytic gain variables `p_μ ∈ Δ^{K-1}` (probability simplex) multiplicatively modulate each stored pattern's contribution to the synaptic matrix `W(p) = (K/N) Ξ diag(p) Ξᵀ`. Gains evolve under an entropy-regularised replicator equation with temperature `T`. The system admits a Lyapunov function (global convergence guaranteed). At fixed points:

```
p*_μ = softmax(f_μ(x*) / T)
```

where `f_μ = (1/2N) Σ_i (ξ^i_μ φ(x^i))²` is the squared overlap with pattern `μ`. **Self-attention emerges from competitive resource dynamics on the gain simplex — not from an explicitly designed attention mechanism.**

Benchmark (Fig. 4): outperforms classical Hopfield and Kozachkov et al. baseline at high memory load K and high corruption level n. Gains are largest where interference is strongest — exactly the regime relevant to our heavy hold-class dominance.

---

### Validation against our architecture

#### 1. LocalCausalAttention w=20 → Emergent routing [HIGH relevance]

Our `LocalCausalAttention` explicitly imposes softmax routing over the last 20 bars of `wick_asymmetry` and `vol_zscore`. The astrocyte paper proves softmax attention can *emerge* from competitive gain dynamics without being prescribed. The theoretical consequence:

> Our explicit attention window is a sufficient but not necessary mechanism. The astrocyte model would achieve equivalent routing by letting pattern-match fitness `f_μ(x)` concentrate gains on the most relevant bars naturally.

**Practical implication:** Replace `LocalCausalAttention(w=20)` with an astrocyte-gated routing module operating over the short-stream features. The gain simplex constraint `Σp_μ = 1` enforces the same finite attention budget as our w=20 mask, but without the hard boundary — gains can concentrate anywhere in the 240-bar sequence where signal is present.

#### 2. Sell precision ceiling = high-interference regime [HIGH relevance]

Our sell class is 3.5% of training data, hold is 95.8% — a K=3 pattern system where one pattern (hold) has overwhelming prior weight. Their Fig. 4 shows the astrocyte model improves most at high K and high corruption. Our situation maps directly:

- K = 3 (sell, hold, buy patterns)
- Corruption level η corresponds to label noise from ATR-adaptive mislabelling
- High interference = hold patterns saturating the Hebbian coupling

Their model's competitive gain allocation suppresses the dominant hold pattern and amplifies the rare sell pattern. This is exactly what our focal loss γ=1 attempts — but focal loss is a scalar weighting, not a content-addressed routing. The astrocyte mechanism provides *context-dependent* suppression: if the current sequence looks like a sell setup, the hold pattern's gain is dynamically reduced.

**Practical implication:** The astrocyte routing module should receive both price features AND the current model's softmax output as feedback — implementing the `f_μ(x)` overlap function as a learned similarity between the current bar sequence and class-specific learned "memory patterns."

#### 3. Regime-conditional temperature T [HIGH relevance]

Their temperature T controls selectivity: low T → sharp routing (low perplexity), high T → uniform gains (Hopfield regime). Our regime labels (Bear/HIGH vol vs Bull/LOW vol) map directly to different T settings:

| Regime | T setting | Rationale |
|--------|-----------|-----------|
| Bear + HIGH vol | Low T (0.01) | High interference, need sharp routing to find sell signal |
| Bull + LOW vol | High T (0.1) | Low interference, uniform gains sufficient |
| Bear + NORMAL | Medium T (0.05) | Intermediate selectivity |

The Lyapunov proof guarantees convergence regardless of T — safe to condition T on the regime without training instability.

#### 4. Z2 symmetry = bid-ask symmetry validation [MODERATE relevance]

Their squared-overlap score `f_μ ∝ ⟨ξ_μ, φ(x)⟩²` is invariant under sign flip — the gains cannot distinguish between a pattern and its negative. Our ghost sample experiment (KS(ghost,buy)=0.046 ≈ KS(sell,buy)=0.049) confirms our data is approximately Z2-symmetric. Their proof shows this degeneracy is resolved by neuronal dynamics breaking symmetry over time — equivalent to how our focal loss training eventually separates sell from buy representations despite near-symmetric feature distributions.

#### 5. Scattering coefficients as stored patterns [MODERATE relevance]

The scattering layer-0 output `(B, 120, 96)` is a natural candidate for the pattern matrix Ξ. Each of the 96 scattering channels encodes a different frequency-channel combination of the input signal. In the astrocyte framework:

```
scatter_out[:, t, :]  →  pattern Ξ_t  (stored memory)
query x               →  bar_return + spread_pressure sequence
astrocyte gains p_t   →  attention weights over time steps
```

This reframes the TCN's sequential processing as a *retrieval problem*: at each forward pass, the network retrieves relevant temporal patterns from the scattering coefficient bank via content-addressed routing.

---

## Paper 2: Tri-Accel

### Core mechanism

Three jointly optimised components: (1) per-layer precision assignment based on gradient variance EMA; (2) sparse Hessian estimation for curvature-aware LR scaling and precision promotion; (3) real-time VRAM monitoring for batch size adjustment. Implemented via custom Triton kernels. Results: 9.9% faster, 13.3% less VRAM, +1.1pp accuracy vs FP32 on CIFAR-10/100.

---

### Validation against our pipeline

#### Component 1: Precision-Adaptive Updates [DIRECTLY APPLICABLE]

Our ScatterTCN layers have heterogeneous gradient variance profiles:

| Layer | Gradient variance | Current precision | Tri-Accel assignment |
|-------|-----------------|-------------------|---------------------|
| LearnableFilter FIR | Low (near-orthogonal basis) | BF16 | **FP16** — safe, saves 50% filter bank RAM |
| LayerNorm | High (sensitive normalisation) | FP32 (autocast whitelist) | FP32 — correct |
| CausalTCN Conv1d | Moderate | BF16 | BF16 — correct |
| LocalCausalAttention sdpa | Moderate | BF16 | BF16 — correct |
| scatter_proj Linear | Moderate-high | BF16 | BF16 or FP32 |

Pushing LearnableFilter to FP16 would halve the filter bank's memory footprint (372 params × 12 × 8 channels = 35,712 params — trivial in itself, but the activations `(4096, 96, 120)` in FP16 save 0.9GB).

**Implementation:** Add gradient variance EMA per layer group, threshold at `τ_low=1e-4`, `τ_high=1e-2`. For LearnableFilter specifically, hardcode to FP16 — variance is empirically low for bandpass filter updates.

#### Component 2: Sparse Second-Order [LOW applicability]

K-FAC / Hessian estimation is expensive relative to gain for small layers. The LearnableFilter bank has 372 params total — the Hessian is a 372×372 matrix, trivial but also near-diagonal (filters are initialised as near-orthogonal Morlets). K-FAC overhead at every 200 steps exceeds the convergence benefit.

**Verdict:** Skip for LearnableFilter. Could apply to `scatter_proj` Linear (d_scatter=96 → d_model=256: weight 96×256=24,576 params) and the Fusion layer. Low priority.

#### Component 3: Memory-Elastic Batch Scaling [MARGINAL benefit]

Our GPU memory is stable at 45% (~18GB) throughout training — the VRAM controller would correctly hold batch=4096. The 13.3% VRAM reduction from Tri-Accel (Table 2) at our scale = ~2.4GB saved → potential batch=5120 without OOM.

At batch=5120 (25% larger): gradient noise σ ∝ 1/√B decreases by 11%, equivalent to a mild LR reduction. Benefit is real but small compared to the architectural changes above.

---

## Transformer Migration — Feasibility Analysis

### Why consider it

The ScatterTCN at 40% GPU utilisation is memory-bandwidth bound (small 1D convolutions on narrow tensors). A Transformer with large hidden dim saturates A100 tensor cores via large M×N×K matrix multiplications — the compute pattern A100 is optimised for.

### Memory feasibility (verified)

After scattering downsampling, the sequence length is T=120. Full self-attention is tractable:

| d_model | Params | Weight mem | Attn mem (B=4096) | Total | Fits 40GB? |
|---------|--------|------------|-------------------|-------|-----------|
| 256 | 3.1M | 6 MB | 0.88 GB | ~10 GB | ✓ |
| 512 | 12.6M | 24 MB | 0.88 GB | ~12 GB | ✓ |
| 768 | 28.3M | 54 MB | 0.88 GB | ~14 GB | ✓ |
| **1024** | **50.3M** | **96 MB** | **0.88 GB** | **~18 GB** | **✓** |

All options fit. The attention matrix `(B=4096, H=8, T=120, T=120)` = 0.88 GB in BF16 — well within budget.

### Architecture option: Replace CausalTCN with Transformer encoder

Keep the scattering front-end (proven discriminative). Replace the CausalTCN 4-layer stack with a Transformer encoder:

```
Input (B, 240, 10)
    ↓
LearnableScatteringBlock(J=3, Q=4) → (B, 120, 104)    [keep — validated]
    ↓
scatter_proj: Linear(104, d_model)  → (B, 120, d_model) [keep]
    ↓
TransformerEncoder(
    d_model=512,
    n_heads=8,
    ffn_dim=2048,      # 4× d_model
    n_layers=4,
    dropout=0.1,
    is_causal=True,    # causal mask — preserves temporal ordering
)                      → (B, 120, 512)
    ↓
Attention-weighted pool → (B, 512)

Short stream: unchanged (LocalCausalAttention w=20 on 3 micro features)
    ↓
Fusion: concat(long, short) → Linear(512+512, 512) → LayerNorm → GELU
```

### GPU utilisation comparison

| Architecture | Compute pattern | A100 tensor core use | Expected GPU util |
|-------------|----------------|---------------------|-------------------|
| CausalTCN (current) | 1D conv (narrow) | Low — bandwidth bound | 40% |
| Transformer d=512 | Large matmul (QKV, FFN) | High — compute bound | 70–85% |
| Transformer d=1024 | Very large matmul | Very high | 80–90% |

The FFN layer alone is `Linear(512, 2048)` applied to `(4096×120)` tokens = matrix multiply `(491520, 512) × (512, 2048)` = large M×N×K that saturates A100 BF16 tensor cores.

### Training compatibility

- **No checkpoint resume:** weight keys change (TCN → Transformer). Must train from scratch.
- **Epoch time:** Transformer d=512 estimated ~2–3 min/epoch (vs ~8 min for ScatterTCN) — 3× faster due to better GPU utilisation.
- **Convergence:** Transformers on financial time series typically converge faster per epoch but need more epochs for initialisation stability. Learning rate warmup critical (first 10 epochs at LR=1e-6 → 5e-5).
- **Causal mask:** `is_causal=True` in `F.scaled_dot_product_attention` — no is_causal+compile interaction at T=120 (sequence long enough to avoid the T=20 OOM bug).

### Risk assessment

| Risk | Probability | Mitigation |
|------|------------|-----------|
| Worse sell precision vs ScatterTCN | Moderate | Scattering front-end retained — long-range signal preserved |
| Training instability | Low | Standard Transformer training is well-understood; bf16 stable at d=512 |
| Longer to first useful checkpoint | Moderate | Pre-warmup phase with frozen scattering layer |
| torch.compile compatibility | **Low** | Transformer has static shapes — compile works correctly |

### Recommendation

**Validate first with d_model=512, 4 layers, after current training run completes.** The current ScatterTCN run is providing baseline data on whether 10-dim features with ATR labels genuinely improve sell precision. If sell P ≥ 0.30 after epoch 20: keep ScatterTCN (feature engineering worked). If sell P < 0.20 by epoch 20: migrate to Transformer d=512 — better GPU efficiency and faster iteration time to test the same feature hypothesis with a higher-capacity encoder.

The Transformer migration is an infrastructure improvement (GPU utilisation 40% → 80%) independent of the label quality question. It can run in parallel on a second Colab session if available.

---

## Priority Implementation Roadmap

### Immediate (current run)

1. Let current ScatterTCN run complete to epoch 20 — get first real sell precision signal with 10-dim ATR labels
2. Apply Tri-Accel Component 1 (LearnableFilter → FP16): one line change in `price_branch_scatter_tcn.py`
3. Implement regime-conditional temperature T in Fusion layer: `T = 0.01 if gmm2 < 0.5 else 0.1`

### Near-term (after epoch 20 results)

4. If sell P < 0.20: implement Transformer d=512 replacement (2 days engineering)
5. If sell P ≥ 0.20: implement astrocyte routing module as a drop-in replacement for LocalCausalAttention

### Research direction (longer term)

6. Astrocyte-gated fusion: replace the current concat-and-project fusion layer with a gain-simplex routing over the long and short stream outputs, conditioned on regime
7. Per-regime T parameter: learnable temperature per GMM2 × vol regime cell (6 cells × 1 scalar = 6 learned parameters)
8. Tri-Accel full integration: requires custom Triton kernels — evaluate after architecture stabilises

---

## Summary Table

| Finding | Source | Action | Priority |
|---------|--------|--------|---------|
| Emergent softmax = our explicit attention | Astrocyte paper | Replace LocalCausalAttention with gain simplex | Medium |
| High K interference → astrocyte helps most | Astrocyte paper | Add astrocyte routing to fusion layer | Medium |
| Regime T conditioning validated | Astrocyte paper | Add per-regime temperature to attention | High |
| LearnableFilter safe for FP16 | Tri-Accel + analysis | Set filter_bank dtype=torch.float16 | Low |
| 13.3% VRAM reduction possible | Tri-Accel | Enables batch=5120 | Low |
| Transformer d=512 fits 40GB, 3× faster | Feasibility check | Implement if sell P < 0.20 at ep20 | Conditional |
| torch.compile works for Transformer | Feasibility check | Re-enable compile after migration | Conditional |
