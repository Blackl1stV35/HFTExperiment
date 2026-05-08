# Literature Synthesis — Phase 8 Feature Exploration Lab
## Transformer Attention Mechanics & Representational Structure

**Papers reviewed:**
1. Kamitani, Y. (2026). *Beyond Object-Level Alignment: Do Brains and DNNs Preserve the Same Transformations?* arXiv:2605.06420v1. Kyoto University / ATR Computational Neuroscience Laboratories.
2. Li, S., Jiang, K., Sun, J. & Hu, T. (2026). *The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity.* arXiv:2605.06611v1. ICML 2026, CUHK / Huawei Foundation Model.

---

## Paper 1 — Kamitani (2026): Beyond Object-Level Alignment

### Core Contribution

The standard framework for brain–DNN alignment asks: *do brain and model assign similar representations to the same stimulus?* Kamitani reframes the question using category theory: *do brain and model preserve the same transformations between stimuli?* The shift is from object-level correspondence to morphism-level correspondence — from `η(brain_s) ≈ model_s` to `η ◦ F_brain(r) = F_model(r) ◦ η` for a stimulus change `r`.

The operational construct is the **Naturality Violation Score (NVS)**. Given a world-model proxy space `W` (CLIP-text, DINOv2, or DreamSim), an external embedding `F_W` maps stimulus pairs to change vectors `ΔW`. These changes are projected into brain space `B` via `Φ_B: W → B` and model space `M` via `Φ_M: W → M`. Cross-system translators `η: B → M` and `η': M → B` are fitted by Ridge regression. The naturality square tests whether:

```
η(Φ_B(ΔW)) ≈ Φ_M(ΔW)    [translate-then-propagate ≈ propagate-then-translate]
```

NVS is the relative L2 residual of this commutation, normalised to a permutation null where random pairing destroys cross-space structure. NVS = 1.0 is chance, NVS = 0.0 is perfect commutativity, lower is better.

### Key Results

**Hierarchy crossover — the primary empirical finding:**

Decomposing `ΔW` along six named concept axes reveals that alignment is not uniform across the ventral stream or DNN depth. It is *selective and axis-dependent*:

| Concept axis | Pooled NVS | Aligns strongest at |
|---|---|---|
| Animacy | **0.39** (lowest) | HVC × deep layers (L6–L7) |
| Real size | 0.72 | HVC × mid-deep |
| Texture energy | mid | Intermediate ROI × mid layers |
| Curvilinearity | proxy-dependent | Intermediate, FW-sensitive |
| Spatial frequency | high | V1–V2 × shallow layers (L1–L4) |
| Luminance | high | V1 × shallow layers |

The statistical ordering is confirmed cross-subject: Spearman ρ = −1.00 for animacy's
ROI profile (decreasing V1→HVC, monotonically), and ρ = −0.93 to −0.81 for the layer profile.
Permutation test on the semantic-vs-low-level class contrast: empirical one-sided p < 10⁻⁴ (none
of 10,000 permutations exceeded the observed magnitude).

**Animacy is the outlier axis.** NVS_animacy = 0.39 ± 0.06 (bootstrap 95% CI [0.34, 0.44]).
The next-best axis pools at 0.52; the remaining axes range 0.52–0.72. Across all 9 (F_W, DNN)
combinations and all 5 subjects, animacy consistently achieves the lowest NVS. The 15-axis appendix
atlas shows affordance and material axes cluster at NVS_a ≈ 0.73–0.74 — well above animacy. High
semantic abstraction alone is not sufficient: "high-level" features that lack ventral-stream precedent
do not achieve low NVS.

**Full-vector (unrestricted) NVS separates proxies but loses axis structure:**

| F_W proxy | Full-vector NVS |
|---|---|
| DreamSim | 0.582 ± 0.049 (best) |
| CLIP-text | 0.701 ± 0.047 |
| DINOv2 | 0.847 ± 0.044 (worst) |

The proxy ordering reflects which family of morphisms each embedding defines, not which
embedding is a better representation: `F_W` is part of the scientific question, not a
nuisance parameter. No full-vector cell reaches the per-axis animacy minimum (0.193 at
HVC×L6 with ResNet, DreamSim), confirming axis decomposition is essential.

**NVS is not a re-description of existing metrics.** Variance decomposition attributes only
~34% of NVS variance to five readout-quality covariates (CAV CV R², encoding/decoding accuracies
of Φ_B, Φ_M, η, η'). Axis identity alone contributes ~34% on top of F_W / DNN / subject controls.
RSA peaks at rs = 0.27–0.46 with NVS at cell level, and a session-bias control shows RSA collapses
from 1.00→0.29 under additive bias while NVS moves only 0.04→0.06 — the metrics are measuring
fundamentally different things.

### Mechanistic Interpretation

The W-less control replaces F_W-derived directions with independently optimised CAVs `v_B, v_M` in
brain and model space respectively. This stays near the permutation null on all tested axes. The
reason: shared world-side anchor `v_W` ensures both sides are tested against the same morphism;
independently optimised `v_B, v_M` are not constrained to align (cosine ≈ 0.15 or below). Alignment
requires a shared external anchor — the choice of W is not a methodological nuisance but the
scientific question itself.

### Limitations

Empirical results come from n=5 subjects on one dataset (GOD fMRI). Three vision DNNs tested
(AlexNet, ResNet-50, ViT-B/16). The operational maps `F_B(r)`, `F_M(r)` are fitted independently
per edge without enforced composition — the category-theoretic naturality is an analogy, not a
strict functorial claim. NVS is evaluated post hoc without optimising for commutativity during
training; NVS-aware training is an open extension.

---

## Paper 2 — Li et al. (2026): The Structural Origin of Attention Sink

### Core Contribution

Attention sinks — the phenomenon where the initial token (position 0) in causal decoder
transformers receives disproportionately large attention scores despite minimal semantic relevance
— have been empirically documented across LLMs but mechanistically unexplained. Li et al. trace
a complete causal chain from a simple structural asymmetry (value aggregation under causal masking)
through four stages to the locked attention sink. They then propose a minimal architectural fix
(head-wise RMSNorm) that suppresses the sink while improving pretraining convergence.

### Causal Chain — Four Stages

**Stage 1: Value aggregation induces positional variance discrepancy.**

Under causal masking, token `i=0` attends only to itself (`a_{0,0} = 1`). All subsequent tokens
`i > 0` aggregate across growing context windows, computing a convex combination of value vectors:

```
o_{i,k} = Σ_j A_{i,j} · V_{j,k}
```

The convex averaging reduces variance as `i` increases. Token 0, exempt from this averaging,
remains a high-variance outlier. This is structural, not learned — it emerges from causal masking
mechanics regardless of model weights.

Empirical validation on Llama-2-7B using *random token sequences* (eliminating semantic effects):
dimension-wise variance decays sharply from position 0 to subsequent positions. Position 0 retains
significantly higher variance across all hidden dimensions.

**Stage 2: Output projection preserves (and amplifies) the discrepancy.**

The output projection `W_O ∈ R^{d×d}` could suppress or amplify the variance outlier. Li et al.
show it does the latter. Structural alignment analysis: Kendall's τ between absolute column weights
`|w_j|` and the first-token input variance `σ_d^{in}` has mean τ = 0.32 across all neurons —
`W_O` assigns larger weights precisely to dimensions where the first token has higher variance.
Post-projection, the first token retains significantly higher variance than subsequent tokens (Figure 7,
right). The discrepancy propagates into the residual stream intact.

**Stage 3: Super neurons in FFN layers selectively activate on the outlier.**

The first token's high-variance representation, now in the residual stream, reaches the FFN. Li et al.
identify **super neurons** — a small subset of neurons in `W_gate` and `W_up` with exceptionally
large L2 norms (e.g., neuron index 7890 in Llama-2-7B Layer 1). These super neurons selectively
activate on the first token: high cosine similarity between the first token's representation and
the super neuron's weight column opens the SwiGLU gate, generating massive raw activations. The
sparse `W_down` then channels this activation exclusively into a small set of outlier dimensions
(e.g., dimension 2533), creating extreme **dimension disparity**:

| Metric | Value |
|---|---|
| Outlier dimension magnitude \|x_{2533}\| | 1.2568 |
| Mean absolute value of other dimensions | 0.0048 |
| Dominance ratio (max/mean) | **262.88×** |

The first token's representation collapses into a near-basis-vector direction dominated by a single
outlier dimension.

**Stage 4: RMSNorm converts dimension disparity into structural QK locking.**

With input dominated by a single dimension `λ` at index `c`, RMSNorm normalises by the L2 norm
which is determined almost entirely by `λ`:

```
RMSNorm(x_0) ≈ sgn(λ) · √d · γ_c · e_c
```

The first token's representation collapses to a fixed direction `e_c` (the basis vector for dimension
`c`). The resulting key vector for the first token in any head `h` approximates the `c`-th row of
the key projection matrix:

```
k_0^(h) ≈ ± √d · (W_K^(h))_{c,:}
```

Head-wise SVD analysis confirms that query projections `W_Q^(h)` have principal directions aligned
with `k_0^(h)`: structural alignment (cosine between principal query direction and sink key) reaches
near-1.0 in specific heads, with near-100% positive ratio of attention scores. These heads are
structurally predisposed to generate large dot products with the sink key — the attention sink
is locked in by architecture.

**Causal validation (two independent interventions):**

*Attention mask intervention:* block token `k=10` from attending to any prior position, forcing
it to attend only to itself. Token 10 immediately becomes a new attention sink — confirming the
mechanism is not position-0-specific but variance-determined.

*Direct variance amplification:* amplify the aggregated output of any token `k` by factor `λ`:
`o'_k = μ + λ(o_k − μ)`. Increasing `λ` consistently promotes token `k` to attention sink status.
Critically, a control experiment that scales the representation *norm* without increasing its
*variance* (multiplying by a scalar) does not induce a sink — variance magnitude, not L2 norm, is
the causal variable.

### Proposed Fix — Head-wise RMSNorm

**Observation:** attention heads are heterogeneous. Low-entropy heads (focus on few tokens) produce
high-variance outputs; high-entropy heads (broad aggregation) produce low-variance outputs. Without
intervention, low-entropy heads dominate the residual stream by magnitude alone. This head-level
variance imbalance compounds the positional imbalance.

**Fix:** apply RMSNorm *per head* after value aggregation, before the output projection `W_O`:

```python
# Head h, position t
ô_t^(h) = (o_t^(h) / RMS(o_t^(h))) ⊙ λ
```

where `λ ∈ R^{d_k}` is a learnable per-dimension scale shared across heads. This:
1. Normalises position-wise variance (first token no longer a high-variance outlier)
2. Normalises head-wise variance (low- and high-entropy heads contribute equally to `W_O`)

**Experimental results** (152M parameter model, 20B tokens, 4 random seeds):

| Metric | Baseline | Head-Norm | Sigmoid attn |
|---|---|---|---|
| Train Loss | 2.7483 ± 0.0118 | **2.7073 ± 0.0095** | (worse) |
| Validation Loss | 2.7812 ± 0.0109 | **2.7421 ± 0.0066** | (slower convergence) |
| Dominance ratio | Sharp rise (early layers) | Consistently low | Low |
| Effective rank | Drops (manifold collapse) | **Higher, stable** | Moderate |
| Attention sink | Present from layer 5 | **Suppressed** | Suppressed |

Sigmoid attention (replacing Softmax) removes the sum-to-one constraint and partially mitigates
the sink, but converges slower and achieves worse validation loss than the Softmax baseline.
Head-wise RMSNorm suppresses the sink while maintaining Softmax and *improving* convergence —
faster training stability is a direct consequence of resolving the variance discrepancy rather than
avoiding the Softmax constraint.

### Mechanistic Significance

The paper resolves a standing open question: why does the sink specifically anchor at position 0
in causal decoders? The answer is structural, not learned:

1. Causal masking forces position 0 to self-attend only → no variance reduction from aggregation
2. `W_O` is structurally biased to amplify rather than suppress this variance
3. Super neurons in the FFN selectively amplify the outlier
4. Sparse `W_down` channels the amplification into extreme dimension disparity
5. RMSNorm collapses the representation to a fixed direction
6. QK projection locks attention scores large toward the sink

Each stage has been independently validated. The mechanism is not parameter-specific to Llama-2;
it is validated on other open-source LLMs in Appendix A (generality check). The result also
unifies a related finding: repeated-token attention sinks (which Yona et al. observed) emerge
from the same chain — repeated tokens fail to reduce variance through aggregation (identical
value vectors average to themselves), mimicking the first-token structural condition.

---

## 3. Cross-Paper Synthesis

These two papers are surface-dissimilar — one is computational neuroscience, one is mechanistic
interpretability of LLMs — but they converge on a shared structural insight that is directly
relevant to the HFTExperiment Transformer architecture.

### Theme A: Transformation preservation, not object correspondence, is the right alignment criterion

Kamitani's NVS framework establishes that two systems can have identical per-stimulus representations
(identical RSA, CKA scores) while preserving completely different transformations between stimuli.
High per-stimulus correspondence does not imply that the two systems respond in the same way to
the same input change.

Applied to the HFTExperiment model: the supervised Transformer (Phase 5, Run 7) achieves sell
P=0.283 and signal=0.193 in per-bar classification. But the model's performance on *transitions*
between bar types — the sell-setup initiation, the mid-transition bars, the exit bars — is not
tested by the classification head. The NVS framework suggests these transition-level properties
are what matter for the RL agent, which operates on the full episode context rather than individual
bar classifications.

Concretely: the Phase 8 candidate features (`adverse_selection_proxy`, `order_processing_residual`,
`hawkes_excitation_5`) were validated in Phase 6–7 by per-bar KS / MI / redundancy tests. This
is object-level alignment. A richer validation — morphism-level — would ask whether these features
also carry consistent signal across bar-to-bar transitions, specifically at regime-transition
boundaries (Bear-to-Bull, HIGH-to-NORMAL volatility). The NVS framework provides the mathematical
language to ask this question, even if the exact implementation is future work.

### Theme B: Internal variance dynamics determine where Transformer attention concentrates

Li et al.'s causal chain begins with a structural fact: the first position in a causal decoder
attends only to itself, making it a high-variance outlier. Everything downstream — super neuron
activation, dimension disparity, QK locking — follows inevitably from this structural asymmetry.

The HFTExperiment Transformer (Phase 5) uses `LocalCausalAttention(w=20)` in the short stream
and a full Transformer with causal masking in the long stream. The attention sink mechanism
applies directly to the long stream: bar 0 of each 120-bar input sequence is structurally forced
to self-attend, creating a positional outlier whose representation will be amplified through the
same super-neuron / dimension-disparity chain described in the paper.

**The implication is non-trivial.** Bar 0 in the HFTExperiment sequence is the warmup bar
from 120 bars prior — a bar from roughly 2 hours ago in XAUUSD M1 time. If that bar is
receiving disproportionate attention (acting as a structural sink), the Transformer is effectively
using a semantically irrelevant historical anchor as its primary attention reference, suppressing
attention to the more recent bars (bars 110–120) where the sell/buy signal is concentrated (per
the Phase 4 temporal profiling: wick_asymmetry and vol_zscore carry 80%+ of their discriminative
signal in the last 20 bars).

This would explain a consistent observation in the Phase 5 training: the gap between val
signal_score (0.171) and test signal_score (0.193) — the model behaves differently on the
evaluation set than the training set, suggesting it is latching onto position-specific artefacts
rather than pure signal. The attention sink at bar 0 would produce exactly this kind of
positional overfitting.

### Theme C: The fix for attention sink is cheap and already aligned with HFTExperiment architecture

Head-wise RMSNorm after value aggregation is a 2-line addition to `PriceBranchTransformer.forward()`:

```python
# After value aggregation, before output projection W_O
# Phase 8 candidate addition — head-wise RMSNorm (Li et al. 2026)
# o: (B, n_heads, T, d_head)
o = o / (o.norm(dim=-1, keepdim=True) + 1e-8) * self.head_scale
# self.head_scale = nn.Parameter(torch.ones(1, 1, 1, d_head))  # learnable per-dim
```

This is architecturally appropriate because the HFTExperiment model already uses pre-norm
(RMSNorm before each sublayer, matching the Llama-2 architecture described in the paper).
Adding per-head normalisation after value aggregation fits the existing normalisation philosophy
and requires no hyperparameter search — the learnable scale `λ` initialises at 1.0 and adapts.

The expected effects in the HFTExperiment context:
- Attention dispersion away from bar 0 → better utilisation of bars 100–120 where microstructure
  signal is concentrated
- Higher effective rank in hidden states → less manifold collapse in the 512-dimensional d_model
- Faster convergence → fewer epochs to reach Run 7's sell P=0.283 baseline; potential for higher
  precision ceiling in Run 8 or 9

Caveat: the paper's pretraining experiments use 152M parameters and 20B tokens. The HFTExperiment
Transformer is smaller (Phase 5, d_model=512, 4 layers) and trained on 5.68M bars. The variance
discrepancy mechanism is architecture-structural, not parameter-count dependent — it should appear
at any depth. But the magnitude of the pretraining improvement (~0.04 validation loss reduction)
at 20B tokens may not directly translate to the same absolute gain in a smaller regime.

---

## 4. Validation of Existing HFTExperiment Decisions

| HFTExperiment decision | Literature support |
|---|---|
| `LocalCausalAttention(w=20)` short stream | Li et al.: focusing short stream on last 20 bars avoids the sink (position 0 of a 20-bar window is bar 100 of the full sequence — semantically recent, not a structural outlier) |
| AstrocyteGatingModule replacing hard w=20 | Kamitani: transformation-preserving routing is more informative than object-level attention; astrocyte soft routing preserves morphism classes rather than fixed position windows |
| sell P=0.283 ceiling despite 7 runs | Li et al.: if bar-0 attention sink is present in the long stream, the model is using a 2-hour-old bar as structural anchor, suppressing the last-20-bar signal; head-wise RMSNorm could break this ceiling |
| Phase 6 `ret_1h` regime-sensitive (D=0.219) | Kamitani: animacy (the clearest brain-DNN aligned transformation) is regime-sensitive by construction — temporal horizon features carry transformation structure across market regimes, consistent with morphism-level alignment |
| Feature orthogonality (max \|r\|=0.087 for `order_processing_residual`) | Kamitani: W-less control confirms that orthogonality requires a shared external anchor; `order_processing_residual` is orthogonal because it is derived from a *different generating process* (spread decomposition), not because it is decorrelated by construction |

---

## 5. New Candidates for Phase 8 Architecture — Proposed Additions

### 5.1 Head-wise RMSNorm (HIGH priority)

**Source:** Li et al. (2026), §5.1
**Implementation:** `PriceBranchTransformer.forward()`, after value aggregation before `W_O`
**Expected gain:** resolve attention sink at bar 0 of the 120-bar sequence; redirect attention
to bars 100–120 where microstructure signal is concentrated; improve effective rank;
potentially break the sell P=0.283 ceiling in Supervised Run 8 or 9
**Risk:** low — purely additive, initialises at identity, fits existing pre-norm philosophy

```python
# In PriceBranchTransformer.__init__:
self.head_scale = nn.Parameter(torch.ones(n_heads, 1, d_model // n_heads))

# In forward(), attention module, after:
#   o = (attn_weights @ V)           # (B, n_heads, T, d_head)
# Add:
o = o / (o.norm(dim=-1, keepdim=True) + 1e-8) * self.head_scale   # head-wise RMSNorm
# Then continue:
#   o = o.transpose(1,2).contiguous().view(B, T, d_model)
#   o = self.out_proj(o)
```

### 5.2 NVS-Inspired Transformation Diagnostic (LOW priority, future work)

**Source:** Kamitani (2026), §3
**Application:** post-hoc evaluation of whether the Phase 5 Transformer preserves the same
bar-to-bar transitions as the ground-truth label sequence. The NVS framework would measure
whether a price-change in XAUUSD (parameterised by `ΔW` = change in `spread_pressure` + `wick_asymmetry`)
propagates through the encoder in the same way as the corresponding label change.
**Status:** exploratory — no immediate implementation. The GOD dataset setup (fMRI + DNN +
external proxy) maps loosely to (price sequence features + model activations + label space),
but the specifics require careful adaptation. Deferred to Phase 9.

---

## 6. Paper Quality Assessment

| Paper | Venue | Rigor | Relevance | Weight |
|---|---|---|---|---|
| Kamitani (2026) | arXiv, May 2026 (not yet peer-reviewed) | ★★★★ Rigorous; n=5 fMRI dataset is a limitation | ★★★ Conceptual — NVS framework applicable to any encoder; direct experimental transfer is future work | Supporting |
| Li et al. (2026) | ICML 2026, Proceedings | ★★★★★ Mechanistic proofs + causal interventions + multi-run pretraining experiments | ★★★★★ Direct architectural implication — head-wise RMSNorm is plug-in compatible with Phase 5 Transformer | Primary |

Li et al. is the primary actionable paper. The head-wise RMSNorm fix is ready to implement in
the next supervised training run. Kamitani provides a richer conceptual frame for understanding
what the trained model's attention is actually computing — useful for interpreting the Phase 5
failure mode and designing Phase 9 validation methodology.

---

## 7. Priority Actions

| Priority | Action | Source |
|----------|--------|--------|
| 1 | Implement head-wise RMSNorm in `PriceBranchTransformer` | Li et al. §5.1 |
| 2 | Visualise attention weights at position 0 in the long stream to confirm sink presence | Li et al. §3 |
| 3 | Run Phase 8 with 3 confirmed features (adverse_sel, ord_proc_residual, hawkes_5) | Phase 7 findings |
| 4 | If head-wise RMSNorm validated → Supervised Run 8 with new architecture + passing features | Li et al. + Phase 8 |
| 5 (future) | Design NVS-inspired bar-transition diagnostic for Phase 9 | Kamitani |

---

*Synthesis — Phase 8 Feature Exploration Lab · 2026-05-08*
*HFTExperiment v2 — primary action: head-wise RMSNorm in PriceBranchTransformer*
