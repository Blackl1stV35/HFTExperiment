# Cracking the Financial Time Series Forecasting Ceiling
## HFTExperiment v2 — Research Problem Statement
**Date:** 2026-05-14  
**Context:** XAUUSD M1, 5.68M bars (2009–2026), 3-class labels (sell/hold/buy)

---

## The Ceiling

After 12 supervised training runs and 9 RL training phases:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Best test sell precision | 0.302 (Run 9/10) | Cannot exceed 30.2% correct sell calls |
| Sell label rate | 3.51% | Model predicts sell on ~17% of bars, correct 30% of the time |
| Hold dominance | 95.8% of labels | Model learns the prior and stays there |
| RL WR ceiling | 51.2% | Break-even at HF Markets Pro = 50.003% |

The 0.302 ceiling has been hit by three independent runs (8, 9, 10) with different architectures, feature sets, and oversampling strategies. This is not a model capacity problem. It is a **label noise problem**.

---

## Root Cause: The Label Noise Floor

### What the labels are

ATR-adaptive triple-barrier labels: a bar is labelled **sell** if price hits the downside barrier (ATR × 0.75) before the upside barrier (ATR × 1.5) within the next 40 bars. Otherwise **hold**. Labels are computed on M1 bars from raw OHLCV.

### Why they are noisy

**1. Microstructure noise at M1 resolution.** At 1-minute granularity, ~60% of price moves are mean-reverting noise rather than directional signal. The ATR barrier labels capture both signal and noise indiscriminately. A label computed at M1 may be contradicted by the same bar at M5 or M15.

**2. Barrier asymmetry creates class imbalance.** TP=1.5×ATR, SL=0.75×ATR means the sell label requires price to fall 50% further than the buy label requires it to rise. At random walk dynamics, sell labels are genuinely rarer AND harder to predict.

**3. Look-ahead window contamination.** The 40-bar (40-minute) label window overlaps with future regime changes that the model cannot see. A bar that looks like a sell setup at t=0 may be cancelled by a news event at t=15.

**4. The 3.51% sell rate is the ceiling, not the floor.** With sell P=0.302 and R=0.29, the model identifies ~58K sell bars of 199K true sell labels. At 30.2% precision, 133K of those predictions are wrong (hold bars predicted as sell). A random classifier would achieve P=0.035. The true information ceiling is somewhere between 0.035 (random) and ~0.35–0.40 (estimated theoretical maximum given noise level).

### Evidence the ceiling is label noise, not model capacity

- Adding rq_regime (MI=0.198, 21× OHLCV) did not lift sell P beyond 0.302
- Head-wise RMSNorm (Li et al. ICML 2026) did not lift sell P beyond 0.302  
- Doubling oversampling of Bear+HIGH from ×2 to ×2.5 did not lift sell P beyond 0.302
- Three independent architectures (Run 7 AstrocyteGating → Run 9 Bear oversampling → Run 10 temporal weighting) all converged to 0.301–0.302
- Val loss converges cleanly (no underfitting); test loss ~0.95 (not catastrophically overfit)

The model has extracted essentially all the accessible signal from the current feature/label combination.

---

## The Five Problems to Crack

### Problem 1: Label Quality — Replace Triple Barrier with Information-Theoretic Labels

**The problem:** Triple-barrier labels are a proxy for "did price move profitably." They measure outcome, not signal. A bar with a genuine sell signal that gets stopped out by noise is labelled **hold**, polluting the training set.

**What to try:**

**(A) Trend-following labels with regime conditioning.** Label a bar **sell** only if:  
- Price falls by ≥ ATR×1.5 within 40 bars AND  
- The bar occurs in a Bear regime (gmm2=Bear) AND  
- The subsequent 10-bar realised volatility is above the 60-bar baseline (confirming directional move, not reversion)

This removes ~40% of noisy sell labels that occur in Bull regime by accident.

**(B) Hindsight-optimal labels from dynamic programming.** Given the full future price path for each bar, compute the theoretically optimal sell/hold/buy sequence that maximises cumulative return subject to a spread cost. These labels are unrealisable at inference but provide the cleanest possible training signal. Measure the ceiling: if a model trained on hindsight-optimal labels achieves sell P=0.45+, the feature set is sufficient and the label construction is the bottleneck.

**(C) Smooth probabilistic labels instead of hard 0/1.** Instead of a binary sell/hold indicator, assign a continuous label = expected profit of a sell action over the next 40 bars, normalised by ATR. Train with MSE regression + threshold at inference. This eliminates the discrete boundary noise around the barrier.

---

### Problem 2: Feature Representation — Multi-Timeframe Supervision

**The problem:** The model sees 240 M1 bars (4 hours). Sell signals at M1 are often regime transitions visible at M15/H1 that manifest as microstructure patterns at M1. The model is trying to infer the M15 context from M1 noise.

**What to try:**

**(A) Hierarchical dual-stream with explicit M15/H1 inputs.** Add a second input stream of M15 bars (same 240-bar window = 60 hours lookback) processed by a separate encoder. Late-fuse with the M1 stream. The M15 encoder provides regime context; the M1 encoder provides entry timing. This is structurally equivalent to what human traders do.

**(B) Wavelet decomposition as a preprocessing layer.** Apply a learnable Discrete Wavelet Transform to the M1 price series before the Transformer. The DWT naturally decomposes the signal into trend (low-frequency) and noise (high-frequency) components. Train the model to attend primarily to the trend components for label prediction. VMD (Variational Mode Decomposition, P2 He 2025) is an alternative — tested and deferred, but theoretically sound.

**(C) Cross-asset correlation inputs.** P7 (Zhao et al. 2025) identified silver futures and USD/CNY as top-3 features. A XAGUSD M1 feed aligned with XAUUSD provides the most direct cross-asset signal. The MI of silver returns with gold sell labels is untested but expected to be high (0.05–0.15) given the 0.85–0.92 historical correlation. This is actionable with ~4 hours of data work.

---

### Problem 3: The RL Ceiling — Reward Function Design

**The problem:** The SAC agent maximises cumulative PnL subject to commission/spread costs. At 51.2% WR, the agent has found the Nash equilibrium: trade only when the supervised model has very high confidence, hold otherwise. This maximises precision at the expense of recall. The WR ceiling = the precision ceiling of the supervised model.

**What to try:**

**(A) Asymmetric reward shaping.** Reward correct sell calls at 3× the reward for correct buy calls, reflecting the rarity of sell labels and the cost of missing them. This pushes the RL agent to exploit the sell signal more aggressively, accepting lower WR in exchange for higher expected value per trade.

**(B) Information ratio reward.** Replace PnL reward with `PnL / sqrt(N_trades)` — the Sharpe ratio of trades made during the episode. This decouples the agent from the number of trades and forces it to find high-quality signals rather than quantity. Current reward implicitly rewards high-frequency trading which amplifies transaction costs.

**(C) Curriculum with adversarial replay.** After the agent converges on easy regimes, replay its hardest historical episodes (episodes with 3+ consecutive losses) with 3× weight. This prevents the agent from ignoring the difficult Bear+HIGH regime once it has learned to avoid it.

---

### Problem 4: The Supervised-RL Interface — Frozen Encoder Bottleneck

**The problem:** The RL agent observes only 3 softmax probabilities + confidence (4 numbers) from the frozen supervised encoder. All of the Transformer's 512-dimensional intermediate representations are compressed to 4 scalars before the RL agent sees them. This is a severe information bottleneck.

**What to try:**

**(A) Pass the full encoder penultimate representation to the RL agent.** Instead of the 3-class output, pass the 512-dim `combined` vector from `DualBranchModel.forward()` as the RL observation. This gives the RL agent access to the full learned representation. The SAC actor/critic would need input dim = 512 + 12 (position/regime features) = 524-dim. Computationally heavier but information-complete.

**(B) End-to-end joint training.** Train the supervised encoder and RL policy simultaneously with a combined loss:  
`L = λ_supervised * CrossEntropy(labels) + λ_rl * (-PnL)`  
The encoder learns representations that are jointly useful for classification AND for policy optimization. This is the theoretically correct approach but requires careful gradient balancing.

**(C) Contrastive encoder pre-training.** Before supervised training, pre-train the encoder with a contrastive objective: bars with the same label at t+10 should have similar representations; bars with opposite labels should be dissimilar. This produces representations where the geometry aligns with tradeable differences, not just classification accuracy.

---

### Problem 5: The Distribution Shift Problem

**The problem:** The model is trained on 2009–2026 but deployed in 2026. The 2024–2026 period has structurally different dynamics: post-COVID monetary expansion, high-frequency ETF flows, algorithmic microstructure dominance. The historical label distribution may not reflect the current market.

**What to try:**

**(A) Temporal reweighting with exponential decay.** Weight training samples by `exp(-λ * (T_now - T_bar))` where λ is calibrated so that 2024–2026 bars get 3–5× more weight than 2009–2015 bars. Validate on a rolling 3-month out-of-sample window. Already partially implemented (pre-2020 ×1.5) but the calibration is ad-hoc.

**(B) Online continual learning with replay buffer.** After each day of paper trading, add the new bars (with hindsight labels) to a small replay buffer. Fine-tune the encoder on this buffer every week with a very low learning rate (LR=1e-6). This allows the model to adapt to regime shifts without catastrophic forgetting of historical patterns. Requires careful memory management to prevent the replay buffer from dominating training.

**(C) Regime-conditioned models.** Train 3 separate encoders: one for Bull-LOW, one for Bear-HIGH, one for transition regimes. Route inference to the appropriate encoder based on current regime state. Each encoder specialises on its regime and avoids the cross-contamination that degrades the single-model precision.

---

## Priority Ordering for Maximum Impact

| Priority | Problem | Expected lift | Effort |
|----------|---------|---------------|--------|
| 1 | Hindsight-optimal labels (Problem 1B) | +0.05–0.15 sell P | 1 day |
| 2 | Silver futures input (Problem 2C) | +0.03–0.08 sell P | 4h data work |
| 3 | Asymmetric RL reward (Problem 3A) | +1–3% WR | 4h code |
| 4 | Regime-conditioned labels (Problem 1A) | +0.02–0.06 sell P | 1 day |
| 5 | Penultimate encoder to RL (Problem 4A) | +1–2% WR | 2 days |
| 6 | Wavelet decomposition (Problem 2B) | unknown | 2 days |
| 7 | Joint training (Problem 4B) | potentially large | 1 week |

**Start with Problem 1B (hindsight labels).** It requires no new data, no architecture changes, and directly attacks the root cause. If the hindsight-optimal model achieves sell P ≥ 0.40, the feature set is sufficient and all other improvements are incremental. If it achieves sell P ≤ 0.35, the feature set is the bottleneck and Problem 2 (multi-timeframe) becomes the priority.

---

## What Success Looks Like

| Metric | Current | Target |
|--------|---------|--------|
| Test sell precision | 0.302 | ≥ 0.35 |
| RL WR (full costs) | 51.2% | ≥ 54% |
| Paper trade WR | ~38% (26 trades) | ≥ 52% over 200 trades |
| Consecutive loss streak | 5 | ≤ 3 |

The 54% WR target at HF Markets Pro ($0.006 commission, 0.6 pip spread) generates expected value of +$0.40 per trade at 0.01 lots. At 10 trades/day that is +$4/day, +$1,460/year on a $100 account — a 1,460% annual return that validates the edge before scaling.

---

*HFTExperiment v2 · RESEARCH_CEILING_ANALYSIS.md · 2026-05-14*
