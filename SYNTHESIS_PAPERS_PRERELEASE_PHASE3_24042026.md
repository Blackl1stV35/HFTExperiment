# XAUUSD v2 — Improvement Note

A review of the system described in `README.md` against ten recent papers. Each
section ties a concrete architectural or training change to a specific module
in the repo, with the paper evidence that motivates it and a rough idea of
effort.

The 10 papers reviewed (shorthand used below in brackets):

- **[Benign-ViT]** Zhang et al. 2026 — *Benign Overfitting in Adversarial Training for Vision Transformers*
- **[CCSO]** Pan et al. CIKM '24 — *Cross-contextual Sequential Optimization via DRL for Algorithmic Trading*
- **[HD-SNN]** Perrinet 2026 — *Working Memory in Recurrent SNNs with Heterogeneous Synaptic Delays*
- **[Sessa]** Horbatko 2026 — *Sessa: Selective State Space Attention*
- **[V-JEPA 2.1]** Mur-Labadia et al. 2026 — *Unlocking Dense Features in Video SSL*
- **[Variance≠Importance]** Salfati 2026 — *Structural Analysis of Transformer Compressibility*
- **[VFA]** Sun et al. 2026 — *Vector Relieved Flash Attention*
- **[Pressure]** Boni 2026 — *Pressure and Generalization: A General Theory of Learning*
- **[Mamba]** Gu & Dao 2023 — *Linear-Time Sequence Modeling with Selective State Spaces*
- **[LeWM]** Maes et al. 2026 — *LeWorldModel: Stable End-to-End JEPA from Pixels*

---

## 1. Encoder — `src/encoder/`

### 1.1 Re-examine the "CNN/TCN over Mamba" decision with Sessa

The README states CNN/TCN was chosen over Mamba SSM because "multi-scale
convolutions handle M1 tick data better at 120-240 bar sequences" and is "4x
faster training". That decision is reasonable for M1 bar data, but **[Sessa]**
changes the landscape: it identifies the exact failure mode your CNN/TCN branch
is also vulnerable to.

**[Sessa]** argues every sequence mixer sits on two axes: *how* routing
coefficients are produced (input-dependent or not) and *how* they are composed
(single-read vs feedback). Under this lens:

- Transformers: input-dependent, single-hop — dilutes as `O(1/S_eff)` when
  attention is diffuse; for a 240-bar window that's the attention spread across
  the whole day.
- Mamba: input-dependent, single-chain multi-hop — decays exponentially in lag
  when "freeze time" fails (noisy price action that never lets `Δ_t → 0`).
- Sessa: many-paths multi-hop via a causal lower-triangular solve
  `(I − B_fb)s = f`. Achieves a power-law `O(ℓ^−β)` memory tail with
  `0 < β < 1`, asymptotically slower than either failure mode above.

For M1 gold with a 120–240 bar context, the relevant long-range events
(overnight gaps, session opens, news echoes) sit right in the regime where
attention is diffuse and SSM decay is exponential. **A Sessa-style mixer is a
natural candidate as a price-branch variant.**

**Recommendation.** Add `src/encoder/price_branch_sessa.py` as an alternative
to the CNN/TCN branch. Keep CNN/TCN as the baseline; A/B them under the same
supervised objective. The key operation is a single strictly-lower-triangular
`torch.linalg.solve_triangular` per layer; still quadratic in `T=240` but tiny
at this length. Worth noting Sessa publishes reference code at
`github.com/LibratioAI/sessa`.

**Effort.** Medium. Two attention heads (forward + feedback), a scalar gain
`γ_t = tanh(⟨a_t, w_γ⟩ + b_γ) ∈ (−1, 1)`, one triangular solve per layer. Sessa
also does *not* use RoPE in the feedback path — the feedback recurrence itself
induces position information, saving parameters.

### 1.2 Fix the attention receptive field using CCSO's Local Attention

**[CCSO]** makes a specific point that applies directly to your price branch
if you go with attention: for 240-step time-series, global attention is both
wasteful and vulnerable to distant-noise bleed. They use a sliding-window
"Local Attention" with a ladder-shaped mask and report that inference
complexity drops from `O(m·h·n²)` to `O(m·h·w²)` where `w` is window size.

Their ablation (their Figure 3) shows performance improves then *degrades*
past a certain window size — too large a window pulls in interfering older
patterns. They settle on window ≈ 20 for their setup.

**Recommendation.** If any attention layer ends up in `price_branch.py` or in
`fusion.py`, use a local/windowed mask with `w ≈ 20–40` M1 bars by default, and
sweep it as a hyperparameter. Do *not* grow it unboundedly with sequence
length.

**Effort.** Low.

### 1.3 Inter-sequence → intra-sequence ordering is correct; formalise it

The README's dual-branch structure already separates price and sentiment
processing. **[CCSO]** validates the two-stage order you'd want *inside* the
price branch: Conv1d first to mix features (inter-sequence, cross-feature),
then self-attention to mix time (intra-sequence). The implicit bet is that
cross-feature correlations change slowly (bid-ask spread × volume relationships
are roughly stable intra-day), while temporal correlations are the fast-moving
signal.

Your current `price_branch.py` does "Multi-scale CNN/TCN + Residual", which is
one-stage. CCSO's two-stage encoder reports meaningfully better long-horizon
performance.

**Recommendation.** Restructure `price_branch.py` as:

```
x → Conv1d(cross-feature) → Residual → LocalAttention(temporal) → output
```

Keep the multi-scale CNN idea; just put it in the first stage with
convolution along the feature axis, not purely the time axis.

**Effort.** Low-medium.

---

## 2. Dense Representations — apply V-JEPA 2.1's insight to SSL pretraining

The README doesn't currently mention SSL pretraining but your `src/training/`
directory has room for it, and this is one of the highest-leverage additions.

**[V-JEPA 2.1]**'s central finding: when you apply a mask-denoising loss only
to masked tokens (the standard V-JEPA 2 / BERT recipe), the visible context
tokens have no incentive to encode *local* structure — they drift toward
behaving like register tokens that aggregate global summaries. The fix is
almost embarrassingly simple: **apply the prediction loss to context tokens
too, not just masked ones.** They call it the Dense Prediction Loss and it
substantially improves dense-feature quality (ADE20K mIoU: 22.2 → 33.9; NYUv2
RMSE: 0.682 → 0.473 — on the *same model*, same compute, just different loss
scope).

Why this matters for XAUUSD. If you do SSL pretraining on M1 bars (mask random
bars, predict their embeddings), the V-JEPA 2.1 failure mode hits you directly:
your context bars drift into "just aggregate the day's state" tokens, and the
fine-grained bar-level representation that you actually need for a 1-minute
trade decision gets washed out. You *care* about bar-level locality — that
entire premise of "M1 tick data" requires it.

**Recommendation.** If you add SSL pretraining — and you probably should — use
a dense predictive loss from day one:

```
L_total = L_predict_masked + L_predict_context   (context term down-weighted)
```

Both losses use the same predictor; the only change is that non-masked context
tokens also contribute to the loss with their own positional supervision.
V-JEPA 2.1 also applies the loss at multiple intermediate encoder layers
("Deep Self-Supervision") for further gains.

**Effort.** Medium if starting from scratch, low if you already have a V-JEPA
style trainer.

---

## 3. Meta-Policy / RL — `src/meta_policy/`

### 3.1 Replace monolithic SAC with CCSO's Probabilistic Dynamic Programming

The README describes "Hierarchical RL, Regime Router → Expert Agents (SAC)".
SAC is continuous-action, but XAUUSD trading in practice reduces to a discrete
action set: long, flat, short. **[CCSO]** points out a clean, underused
advantage of the discrete case: you can compute the *exact expected return* at
each timestep by summing over all possible actions, rather than sampling one.

Their gradient is:

```
∇_θ L(s_t) = Σ_a [F(s_t, a) + H(s_t, a) − E(G(s_t))] · ∇_θ π(a|s_t)
```

where `F` is the immediate reward expectation, `H` is the discounted
continuation value, and the `E(G(s_t))` baseline subtraction reduces variance
(like REINFORCE with baseline, but exact rather than estimated). With a
three-action space this is three forward passes per timestep instead of the
policy-gradient Monte Carlo rollout SAC uses. CCSO report considerably better
learning stability and, crucially, **they are deployed in production on CSI
1000.**

**Recommendation.** For the part of the policy that produces long/flat/short,
replace SAC with PDP-style gradient computation. Keep SAC (or an equivalent
continuous-action policy) only for the position-sizing head if that head is
continuous. File: `src/meta_policy/rl_agent.py`.

**Effort.** Medium. The main work is rewriting the learning step; the policy
network stays the same.

### 3.2 Curriculum learning on transaction costs

**[CCSO]** also introduces something quietly important: they anneal
transaction fees during training. Specifically, they split training into
evolves (they use 10), with the trading fee rate scaled linearly from 0 at
evolve 1 to the final value at evolve 10. Their deployment analysis shows
different evolves behave *qualitatively* differently — low-evolve models are
more active and capture smaller fluctuations, high-evolve models are more
cautious — and in production they group stocks by volatility and route them to
different evolve-stage models.

Your README mentions circuit breakers but not cost curriculum. This is
directly applicable to XAUUSD where overnight holds and partial fills create
real slippage.

**Recommendation.** In `src/training/train_rl.py`, schedule spread/commission:
zero at epoch 0, real values by epoch N (cosine or linear warmup). Keep several
evolve-stage checkpoints (not just the final one) and consider serving
different checkpoints based on current market volatility regime — this is the
most deployment-ready idea from CCSO.

**Effort.** Low — a few lines plus checkpoint retention policy.

---

## 4. GAN Market Simulation — `src/meta_policy/gan_market.py`

Your README specifies "GAN-Simulated Market for RL Training". GANs are a
reasonable choice for market simulation but there are two warnings and one
alternative from the papers.

### 4.1 Watch for V-JEPA-style mode collapse on "context tokens"

If your GAN emits full OHLCV sequences and the RL agent only conditions on a
subset (e.g. masked-future scenarios), you'll hit the **[V-JEPA 2.1]** failure
mode in reverse: the generator will learn to produce plausible-looking
*aggregate* statistics while being lazy about individual bar-level dynamics,
because the discriminator's only signal comes from the aggregate distribution.

**Recommendation.** Add a bar-level consistency term to the GAN loss —
something that penalises disagreement between the generator's predicted
local statistics (return kurtosis, realised vol on 5-bar windows) and those
of real data, not just the full-sequence statistics.

### 4.2 Consider a JEPA world model in latent space as a cheaper alternative

**[LeWM]** is worth considering as a replacement or complement to the GAN.
LeWM trains stably with **two loss terms** (next-embedding prediction + SIGReg
Gaussianity regulariser on latents), **15M parameters, single GPU, a few
hours** — and plans 48× faster than a foundation-model-based world model at
matched or better control performance.

For your setting, the substitution looks like:

- Encoder: your dual-branch encoder → latent `z_t` (already doing this).
- Predictor: a small transformer that predicts `z_{t+1}` from `z_t` and the
  action `a_t` (long/flat/short).
- Regulariser: SIGReg — project latents onto `M=1024` random directions and
  apply an Epps–Pulley normality test on each projection. By Cramér–Wold, if
  all marginals are Gaussian, the joint is Gaussian. This is the only anti-
  collapse mechanism and **they have provable anti-collapse guarantees** — GANs
  famously do not.
- Planning: Cross-Entropy Method (CEM) in latent space to optimise
  `a_{1:H}` minimising `‖ẑ_H − z_goal‖²`. For trading the "goal" becomes
  maximum expected return at horizon H rather than a target embedding, but the
  machinery is the same.

This gives you model-based RL with a planner you actually understand, for far
less compute than the GAN approach.

**Recommendation.** Add `src/meta_policy/lewm_world.py` as a parallel track.
Benchmark it against the GAN on (a) wall-clock training cost, (b) downstream
policy quality, (c) stability across seeds. Honestly, I'd expect LeWM to win on
at least two of those.

**Effort.** Medium. SIGReg is not hard to implement (there's reference code at
the LeWM repo). The bigger cost is plumbing CEM into your planning loop.

---

## 5. Confidence Head — `src/encoder/fusion.py`

The README has a confidence head feeding position sizing. Two specific
improvements from the papers:

### 5.1 Don't use activation variance as the confidence signal

**[Variance≠Importance]** is a sharp empirical result that matters here: on
GPT-2 and Mistral 7B, high-variance activation directions are **96%
uncorrelated** with directions that actually predict the output (measured by
CCA vs PCA). Projecting onto the top-8 PCA directions (capturing 95% of
variance) blows perplexity from 47 to 3,441.

If your current confidence head is something like "softmax entropy" or
"norm of the fused representation", you're probably picking up *variance*
rather than *importance*. The paper's recommendation — "use downstream-aware
importance metrics" — maps directly to: train confidence against held-out
profitability, not against representation statistics.

**Recommendation.** Train confidence as a separate prediction head that
regresses the *realised Sharpe over the next N bars* given the current
embedding, then calibrate it via isotonic regression on a validation split. Do
not tie confidence to activation magnitude or representation entropy. Use this
calibrated confidence — not a raw logit — for position sizing.

**Effort.** Low-medium. Mostly a change to the training objective for one
head.

### 5.2 The four-phase architecture insight → where to read confidence from

**[Variance≠Importance]** also identifies a clear four-phase structure in
transformer depth: (1) context building, (2) feature construction, (3)
refinement, (4) prediction assembly. On Mistral 7B, block-level linearity
grows from R² = 0.17 at block 0 to R² = 0.93 at block 31. Early blocks do
nonlinear feature construction; late blocks do near-linear refinement.

For your fusion head this suggests: **read confidence from an intermediate
layer, not the final one**. The final layer in a well-trained decoder is in
"linear refinement" mode — near-deterministic given its input distribution,
which makes its activations a poor signal for epistemic uncertainty. The
intermediate "feature construction" layers carry more information about when
the model is in unfamiliar territory.

**Recommendation.** Tap the confidence head from an intermediate encoder
layer (empirically, around the 60–70% depth point). Combine with the
calibrated regression head from §5.1.

**Effort.** Low.

---

## 6. Training Stability — `src/training/`

### 6.1 Adversarial training, if used, has a known benign regime

The README doesn't currently mention adversarial training, but if you're
considering it for robustness (a reasonable thought given adversarial price
movements in thin-liquidity conditions), **[Benign-ViT]** gives you the exact
recipe.

Their Theorem 4.2 is worth internalising: adversarial training on ViTs is in
one of three regimes depending on perturbation budget `τ` and signal strength
`‖μ‖₂`:

1. **`τ ≤ O(‖μ‖₂ / log d_h)`** — attention learns normally, robust benign
   overfitting with clean test error `exp(−C·d·SNR²)`.
2. **`O(‖μ‖₂/log d_h) ≤ τ ≤ O(‖μ‖₂)`** — attention weights collapse to
   uniform, ViT degenerates to a linear model. Still converges but M times
   slower.
3. **`τ ≥ ‖μ‖₂`** — robust test error `≥ 0.25`. No amount of training helps.

The practical takeaway: **if you do adversarial training, bound `τ` below
`‖μ‖₂ / log d_h`, and verify the signal-to-noise ratio empirically satisfies
`N · SNR² = Ω(1)`.** Otherwise you're either producing a glorified linear
model or actively degrading generalisation.

**Recommendation.** If you add adversarial augmentation (e.g. small random
perturbations of price inputs to simulate slippage/quote noise), keep
perturbation magnitude small and log `‖μ‖₂` vs `τ` throughout training. In
file `src/training/train_supervised.py`.

**Effort.** Low if only used for augmentation; this is more of a "don't
accidentally enter regime 2 or 3" warning than a positive recommendation.

### 6.2 Treat regularisation as the mechanism of generalisation, not a safeguard

**[Pressure]** is speculative by ML-paper standards, but its central
empirical finding is not: on modular arithmetic, weight decay = 0 produced
zero generalisation across *hundreds of runs* at every architecture and size
tested. Adding weight decay reliably produced generalisation. The paper
proposes a general theory where "pressure" (weight decay, dropout, data
augmentation, capacity restriction) is the mechanism that drives the
transition from memorisation to generalisation, with a viable range — too
little fails, too much catastrophically collapses.

Two useful operational points from this paper:

1. **Composition**: two sub-threshold forms of pressure, neither sufficient
   alone, together cross the threshold. Their example: weight decay alone and
   data augmentation alone both failed; together they produced full
   generalisation.
2. **Scaling**: they find `P_opt ≈ 0.47 · h^0.31` for modular arithmetic with
   weight decay. The exact exponent doesn't transfer to your setting, but the
   sub-linear scaling of optimal regularisation with model width does — it
   means you need *more* weight decay than you'd naively estimate when you
   scale the encoder.

The README mentions dropout (probability 0.37 in one test) but not a principled
regularisation strategy. Dropout alone on a 240-bar sequence model, at the
scale you're likely training, is probably below the viable threshold.

**Recommendation.** Stack at least three distinct forms of pressure:

- weight decay (AdamW, 1e-2 to 1e-4, log-sweep)
- dropout in fusion (0.1–0.3)
- input augmentation: add small synthetic jumps, time warps, or masked bars
  during training (effectively your GAN scenarios sampled at low strength)

Sweep them together with optuna/Bayesian search, not independently. In
`src/training/train_supervised.py`.

**Effort.** Low.

---

## 7. Memory of Regimes — `src/meta_policy/` (Regime Router)

**[HD-SNN]** won't go into production as an SNN, but its core insight
transfers cleanly to your regime router. Perrinet shows that heterogeneous
synaptic *delays* give a recurrent network working memory for arbitrary-
length spike sequences, with key findings:

- Delay depth `D` is the primary capacity lever: capacity scales as `N² × D`
  via `N × D × p_A` context orthogonality.
- Pattern duration drives compounding error: `loss ∝ T`.
- Sparse activity (`p_A ≈ 10⁻³`) gives the best context orthogonality; too
  dense and contexts interfere.

The analogue for your regime router is a **delay-line memory** rather than an
RNN hidden state: instead of trying to summarise market state into a single
vector that a GRU/LSTM maintains, hold a buffer of the last D minutes' events
(tagged by type: volume spike, spread widening, directional break), and let
the regime router attend across the *delay axis* rather than the time axis.

This is mechanically similar to what CCSO's Local Attention already does, but
with a specific argument for *depth* (long D) over *width* (more neurons): the
same memory-capacity scaling argument from **[HD-SNN]** says you get more
working memory from deeper delay windows than from larger hidden state.

**Recommendation.** In `src/meta_policy/regime_router.py` (not currently
listed but implied), implement a delay-line buffer of tagged events with
D ≈ 60–240 minutes, with sparse triggering. Use cross-attention from the
current bar to this buffer instead of a GRU state. This maps directly to
**[Mamba]**'s selective-SSM framing too (see §8) but with a crisper capacity
argument.

**Effort.** Medium. Depends on how you currently implement the router.

---

## 8. Mamba as a drop-in for the regime router

**[Mamba]** is the canonical "content-based selection via SSM" architecture,
and the selective-copying / induction-heads story maps cleanly onto regime
routing:

- You want the router to *remember* that a specific pattern (e.g. Asian
  session gap) happened hours ago, but only when the current state indicates
  that memory is relevant.
- The `Δ_t` parameter in Mamba's selective SSM is the exact mechanism: small
  `Δ_t` = "don't forget", large `Δ_t` = "reset / forget", both being
  input-dependent.
- The 5× inference throughput over transformers at matched quality is real and
  directly relevant to your "below 10ms per timestep" requirement from the
  LSTM decoder.

**Caveat from [Sessa]**: Mamba's "failed freeze-time" regime is a real
failure mode under noisy conditions — if market noise is high enough that the
model cannot create a long corridor of `Δ_t ≈ 0`, long-range memory decays
exponentially. This is exactly the situation in volatile XAUUSD sessions.

**Recommendation.** Treat Mamba and Sessa as two candidates with opposite
failure modes. Mamba is the efficiency winner in calm markets; Sessa is the
robustness winner in diffuse/noisy regimes. Given your HITL setup, you can
afford to A/B them conditionally on a volatility estimate. File
`src/meta_policy/regime_router.py`; the selective-SSM math is standard by now
and reference implementations are solid.

**Effort.** Low-medium for Mamba (mature libs exist), medium for Sessa.

---

## 9. Inference Latency — `src/inference/` and `rust_inference/`

### 9.1 VFA for the attention kernel

If your inference path uses standard FlashAttention, **[VFA]** is worth
pulling in. Their claim is precise: when tensor cores are saturated, the
non-matmul components of online softmax (per-tile `rowmax` reductions, rescale
chain) become vector/SIMD-limited and dominate latency. VFA addresses this via
three steps:

1. `m`-initialisation from cheap key-block representations (avoids starting
   from `-∞`).
2. Sink + local reordering of the key-block scan.
3. Max-freezing on non-sink, non-local blocks — skipping per-block `rowmax`
   and the rescale chain.

They report **~2× speedup on C8V32/C4V32/C4V16 configs** with **~0% accuracy
change** on MMLU/MATH/HumanEval. Crucially they compose with block-sparse
attention (BLASST), giving VSA, for multiplicative gains.

Whether this matters to you depends on your inference hardware. If you're on
GPU with long context, yes. If you're on CPU-only edge for order routing, no —
the bottleneck is elsewhere. Given the README mentions a Rust inference engine
and likely runs on reasonable hardware, VFA is worth a look.

**Recommendation.** If inference latency matters more than slight numerical
deviation, implement VFA in the Rust inference kernel, or adopt FA4 which has
VFA-like rescale-elision as a conditional path. File: `rust_inference/`.

**Effort.** Medium-high. Kernel work is kernel work.

### 9.2 Adaptive per-token computation (30% of tokens are easy)

**[Variance≠Importance]** reports that **~30% of tokens are computationally
easy** (measured independently via trained exit heads and via KL sensitivity
analysis), suggesting adaptive per-token computation as a better investment
than static compression.

For trading, this maps to: **~30% of market states are "obvious holds" or
"obvious flats"** where the full encoder-decoder is overkill. If you can
cheaply detect these and short-circuit to a default action, you save latency
for the states that matter.

**Recommendation.** Train a small auxiliary "confidence/triviality" head at
an early layer of the encoder (say after the price branch, before fusion).
At inference, if confidence on "flat position" exceeds a threshold, skip
fusion and meta-policy and emit flat directly. Gate this behind HITL — you
don't want the short-circuit firing on genuinely unusual conditions. File:
`src/inference/`.

**Effort.** Medium. The paper notes that *post-hoc* early exit heads had a
poor quality-speed tradeoff (+4.31 PPL for 6.4% compute saved); training the
model to expose exit points improves this substantially but requires an
integrated training objective with `L = L_final + 0.5 · L_early`.

---

## 10. A few things *not* to do

Some of these papers point to easy-looking wins that are actually traps:

- **Don't use rank reduction / SVD-based compression on the encoder.**
  **[Variance≠Importance]** is clear: factored weight approximations amplify
  errors through cross-terms. Direct INT4 quantisation dominates rank-`k`
  projection by 2–25× MSE at the same bit budget. For your `inference/`
  deployment, quantise; don't factor.

- **Don't replace multiple consecutive blocks with linear approximations.**
  Same paper: single-block replacement works (R² ≈ 0.95, +1.71 PPL on
  Mistral), but multi-block replacement fails catastrophically (R² → 0.60
  across 5 blocks, +151 PPL). The reason is distribution shift cascading
  through residual connections — a direct analog to your sequential
  inter-minute dependencies.

- **Don't over-rotate your inputs.** PCA/CCA rotations looked promising on
  paper for compression but produced 20–21× worse MSE than direct INT4. The
  input geometry already is the geometry the model was trained on; don't
  second-guess it.

- **Don't assume a longer attention window is always better.** CCSO's own
  ablation (their Figure 3) shows performance peaks then degrades as window
  size grows beyond the regime-typical pattern length.

---

## Priority ordering

If you only do three things from this note, do them in this order:

1. **Calibrate confidence against realised Sharpe** (§5.1). Smallest code
   change, biggest impact on HITL trust and on position sizing quality.
2. **Curriculum on transaction costs** (§3.2). Also small, and gives you a
   ladder of checkpoints you can route between based on volatility regime.
3. **Replace GAN market sim with LeWM** (§4.2). Bigger lift, but changes the
   asymptotic compute cost of improving your model from "expensive" to
   "manageable", and comes with anti-collapse guarantees the GAN does not.

After that, the Sessa/Mamba A/B (§1.1 + §8) and the dense predictive loss
(§2) are the next tier. VFA (§9.1) is last unless you're already latency-
limited in inference.
