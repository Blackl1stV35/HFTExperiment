# PAPER_TRADE_UPDATE_09052026.md
**Date:** 2026-05-09  
**Phase:** Paper Trade Preparation — HFTExperiment v2  
**Account:** HF Markets **Demo Premium** (MT5, $100,000 virtual, 60-day expiry)  
**Checkpoint:** `rl_agent_evolve2.pt` (Phase 8b) + `dual_branch_last.pt` (Run 10)

---

## Account Selection

**Use Demo Premium** — not Demo Contest, not Demo Zero.

- **Demo Premium** mirrors the real **Pro account** cost structure (0.6 pip spread, no commission) — the same cost model the RL curriculum was trained against (evolve 2 = comm=$0.14, spread=0.4pip). Demo Contest uses fixed $10,000 balance with contest rules that don't match live conditions. Demo Zero has commission (unknown amount, different cost structure).
- $100,000 virtual balance with adjustable leverage — use 1:100 for paper trading to simulate a $1,000 real account at 0.01 lot per trade.
- **60-day expiry** is sufficient for the 2–4 week validation window.

---

## What Needs to Be Updated in the Local Pipeline

The existing `paper_trade.py` has the correct skeleton (MT5 connection, circuit breaker, HITL gate, Telegram alerts) but is missing the Phase 8b inference stack. Seven changes required:

---

### 1. `scripts/paper_trade.py` — Replace inference engine with PyTorch + RL layer

**Current:** Uses `ONNXInferenceEngine` reading `exports/best_model.onnx`. Action = argmax of supervised model output directly.

**Replace with:**
```python
# At top of file — replace ONNXInferenceEngine import with:
import torch
from src.encoder.fusion import DualBranchModel
from src.meta_policy.rl_agent import ConfidenceSACAgent

# In TradingLoop.__init__():
self.supervised_model = None
self.rl_agent         = None
self.bar_buffer       = []    # (N, 6) raw OHLCV + spread, max 300 bars
self.scaler_state     = {}    # rolling RobustScaler state per feature

# In TradingLoop.initialize():
# Load supervised model
sup_ckpt = self.config["inference"]["supervised_checkpoint"]
self.supervised_model = DualBranchModel(cfg)
self.supervised_model.load_state_dict(
    torch.load(sup_ckpt, map_location="cpu")["model_state_dict"]
)
self.supervised_model.eval()

# Load RL agent
rl_ckpt_path = self.config["inference"]["rl_checkpoint"]
self.rl_agent = ConfidenceSACAgent(obs_dim=16, hidden_dims=[512, 512], device="cpu")
self.rl_agent.load(rl_ckpt_path)
```

---

### 2. `scripts/paper_trade.py` — Replace 6-dim feature builder with 10-dim

**Current:** Builds `[bid, bid+0.5, bid-0.5, bid, 100, spread]` — placeholder only.

**Replace the feature computation in `run()` loop with:**
```python
def _build_features(self, bars: list[dict]) -> np.ndarray | None:
    """Convert raw M1 bars to 10-dim normalised feature array (240 bars)."""
    if len(bars) < 240:
        return None
    bars240 = bars[-240:]
    
    opens  = np.array([b["open"]  for b in bars240], dtype=np.float64)
    highs  = np.array([b["high"]  for b in bars240], dtype=np.float64)
    lows   = np.array([b["low"]   for b in bars240], dtype=np.float64)
    closes = np.array([b["close"] for b in bars240], dtype=np.float64)
    vols   = np.array([b["tick_volume"] for b in bars240], dtype=np.float64)
    spreads = np.array([b["spread"] for b in bars240], dtype=np.float64)

    # bar_return_bps: log return in basis points
    bar_ret = np.concatenate([[0.0], np.log(closes[1:]/(closes[:-1]+1e-8))]) * 10000

    # wick_asymmetry: (high-close)/(high-low) - (close-low)/(high-low)
    hl = highs - lows + 1e-8
    wick_asym = (highs - closes) / hl - (closes - lows) / hl

    # vol_zscore: tanh(z-score of tick_volume over 20 bars)
    vol_zscore = np.zeros(240, dtype=np.float32)
    for i in range(240):
        lo = max(0, i - 19)
        w  = vols[lo:i+1]
        mu, sig = w.mean(), w.std()
        vol_zscore[i] = float(np.tanh((vols[i] - mu) / (sig + 1e-8)))

    # spread_pressure: spread × vol_zscore
    spread_norm  = (spreads - spreads.mean()) / (spreads.std() + 1e-8)
    spread_press = np.tanh(spread_norm * vol_zscore)

    # Stack 10 features
    raw = np.stack([opens, highs, lows, closes, vols, spreads,
                    bar_ret, wick_asym, vol_zscore, spread_press], axis=1).astype(np.float32)

    # RobustScale per-feature (approximate — use rolling median/IQR)
    for j in range(10):
        col = raw[:, j]
        med = np.median(col); iqr = np.percentile(col, 75) - np.percentile(col, 25) + 1e-8
        raw[:, j] = np.clip((col - med) / iqr, -3, 3)

    return raw  # (240, 10)
```

The bar buffer needs M1 OHLCV bars from MT5. Replace tick accumulation with M1 bar polling:
```python
# In run() loop — replace tick-based logic with M1 bar polling:
import MetaTrader5 as mt5
bars = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 245)
# bars is a numpy structured array with fields: time, open, high, low, close, tick_volume, spread
```

---

### 3. `scripts/paper_trade.py` — Add regime deploy gate

**Add before any trade execution:**
```python
def _get_regime(self, bars: list) -> tuple[float, float]:
    """Approximate GMM2 and vol_enc from recent bars.
    Returns (gmm2_state, vol_enc) where gmm2: 0=Bear 1=Bull, vol_enc: 0=LOW 1=HIGH.
    Simple heuristic: use 20-bar return sign for GMM2, ATR ratio for vol.
    """
    closes = np.array([b["close"] for b in bars[-25:]])
    ret20  = (closes[-1] - closes[-20]) / closes[-20]
    gmm2   = 1.0 if ret20 >= 0 else 0.0  # Bull if positive 20-bar return

    highs  = np.array([b["high"] for b in bars[-15:]])
    lows   = np.array([b["low"]  for b in bars[-15:]])
    atr14  = np.mean(highs - lows)
    atr_baseline = np.mean([b["high"]-b["low"] for b in bars[-60:-15]])
    vol_enc = 1.0 if atr14 > atr_baseline * 1.4 else 0.0  # HIGH if ATR elevated

    return gmm2, vol_enc

# Deploy gate in run() loop:
gmm2, vol_enc = self._get_regime(self.bar_history)
if gmm2 == 0.0 and vol_enc == 1.0:
    logger.info("Deploy gate: Bear+HIGH regime — skipping")
    continue
```

---

### 4. `scripts/paper_trade.py` — Build 16-dim RL obs and get RL action

**Replace the `action = int(probs.argmax())` block with:**
```python
# After computing supervised probs and confidence:
# 1. Build RL obs (16-dim)
def _build_rl_obs(self, probs, conf, position_dir, unrealized_pnl,
                  hold_frac, atr_norm, trend_norm, session_phase,
                  regime_quality, gs_quartile, cu_au_regime,
                  ret_1h, ret_15m) -> np.ndarray:
    return np.array([
        probs[0], probs[1], probs[2], conf,          # 0-3
        float(position_dir), unrealized_pnl, hold_frac, # 4-6
        atr_norm, trend_norm, session_phase,          # 7-9
        regime_quality, gs_quartile, cu_au_regime,    # 10-12
        ret_1h, ret_15m,                              # 13-14 (Phase 6)
        0.0,                                          # 15 VIO disabled
    ], dtype=np.float32)

# 2. Get RL action (deterministic — use actor mean, no sampling)
obs_tensor = torch.FloatTensor(rl_obs).unsqueeze(0)
with torch.no_grad():
    action_raw = self.rl_agent.actor(obs_tensor).cpu().numpy()[0]
# action_raw: [position_size_signal, exit_signal] — interpret:
# position_size_signal > 0.3 AND supervised action != hold → enter
# exit_signal > 0.5 AND position open → exit
```

For ret_1h and ret_15m, compute from the bar history:
```python
def _mtf_return(self, closes: np.ndarray, window: int) -> float:
    """Last bar's z-scored cumulative log-return over `window` bars."""
    if len(closes) < window + 10:
        return 0.0
    log_rets = np.log(closes[1:] / (closes[:-1] + 1e-8))
    rolling  = np.convolve(log_rets, np.ones(window), mode='same')
    w        = rolling[-120:]
    return float(np.tanh((rolling[-1] - w.mean()) / (w.std() + 1e-8)))
```

---

### 5. `configs/deployment/production.yaml` — Add RL and deploy gate fields

```yaml
inference:
  engine: pytorch          # changed from onnx
  device: cpu
  supervised_checkpoint: models/dual_branch_last.pt    # Run 10 ep61
  rl_checkpoint: models/rl_agent_evolve2.pt             # Phase 8b evolve2
  max_latency_ms: 10.0     # relaxed — PyTorch CPU ~3-5ms

deploy_gate:
  enabled: true
  block_bear_high: true    # Do not trade when gmm2==Bear AND vol==HIGH
  regime_lookback_bars: 60 # bars used for regime estimation

broker:
  type: mt5
  symbol: XAUUSD
  lot_size: 0.01           # 0.01 lot = $1/pip on XAUUSD — safe for $100k demo
  max_lots: 0.01           # single position, no scaling during paper trade
  magic_number: 20260509
```

---

### 6. Copy model files to `models/` directory

```
models/
├── dual_branch_last.pt          ← from Google Drive (Run 10 ep61)
└── rl_agent_evolve2.pt          ← from Google Drive (Phase 8b rl_checkpoints_p8/)
```

Download from Colab:
```python
# In Colab:
import shutil
shutil.copy("/content/drive/MyDrive/Colab Notebooks/dual_branch_last.pt",
            "/content/HFTExperiment/models/dual_branch_last.pt")
shutil.copy("/content/drive/MyDrive/Colab Notebooks/rl_checkpoints_p8/rl_agent_evolve2.pt",
            "/content/HFTExperiment/models/rl_agent_evolve2.pt")
```
Then sync to local machine via Google Drive or direct download.

---

### 7. Test synthetic mode first

```bash
# Windows command prompt, from project root:
python scripts/paper_trade.py --config configs/deployment/production.yaml --synthetic
```

Should run without errors, printing signal logs every 50 ticks. Then switch to live Demo Premium:
```bash
python scripts/paper_trade.py --config configs/deployment/production.yaml
```
MT5 login/password/server from your HF Markets demo account email.

---

## Paper Trade Validation Checklist (2–4 weeks)

Track these metrics daily in a CSV log:

| Metric | Target | Stop if |
|--------|--------|---------|
| Win rate (closed trades) | ≥ 50.5% | < 48% for 5 consecutive days |
| Trades per day | 5–25 | > 50 (overtrading) or 0 (gate blocking all) |
| Bear+HIGH gate activation | 10–20% of bars | > 50% (regime detection broken) |
| Max daily drawdown | < 1% | > 2% (circuit breaker fires) |
| Avg trade duration | 30–120 min | < 5 min (noise trading) |

If WR ≥ 50.5% over 200+ trades → proceed to real Pro account ($100 minimum).

---

## Final Training Report Summary

| Metric | Value |
|--------|-------|
| Supervised model | Run 10, ep61 |
| Test sell precision | 0.301 |
| Test buy precision | 0.160 |
| Test signal score | 0.162 |
| RL phase | Phase 8b |
| RL obs dimensions | 16 (VIO disabled) |
| RL WR at full curriculum costs ($0.70 + 2pip) | **51.2%** |
| RL WR at HF Markets Pro real costs (~$0.006/trade) | **~52–54% est.** |
| Deployment checkpoint | `rl_agent_evolve2.pt` |
| Deploy gate | Bear+HIGH regime → no trade |
| Total training runs | 10 supervised + 8 RL phases |
| Training data | 5,680,771 XAUUSD M1 bars (2009–2026) |

---

*HFTExperiment v2 · Paper trade setup guide · 2026-05-09*
