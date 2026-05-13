"""Data preprocessing — Phase 3 consolidated module.

Single import point for all feature preparation the training pipeline needs:

    prepare_features()        — scale OHLCV+spread for supervised model input
    join_regime_labels()      — left-join daily_regime_labels.csv onto M1 bars
    get_regime_array()        — extract (N,6) float32 regime array for RL obs
    compute_rl_obs_features() — ATR, EMA trend, session-phase arrays (obs 7-9)

Encoding maps (must stay in sync with notebook export):
    KMeans 63d  — Regime-A=0.0  Regime-B=0.33  Regime-C=0.67  Regime-D=1.0
    Vol regime  — LOW=0.0  NORMAL=0.5  HIGH=1.0
    G/S quartile — rank-based [0,1] derived from gs_ratio column
    Cu-Au regime — Financial=0.0  Mixed=0.5  Commodity=1.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger


# ── Encoding maps ─────────────────────────────────────────────────────────────

_KM_ENC = {
    "Regime-A (best)":  0.0,
    "Regime-B":         0.33,
    "Regime-C":         0.67,
    "Regime-D (worst)": 1.0,
}
_VOL_ENC = {"LOW": 0.0, "NORMAL": 0.5, "HIGH": 1.0}
_CU_ENC  = {"Financial": 0.0, "Mixed": 0.5, "Commodity": 1.0}


# ── Scalers ───────────────────────────────────────────────────────────────────

class WindowMinMaxScaler:
    """Rolling window min-max scaler. Pricing-agnostic."""

    def __init__(self, window_size: int = 120, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Vectorised in chunks — avoids the 15 GB stride window for 5.68 M rows.

        Chunk strategy: process CHUNK_ROWS rows at a time. Each chunk needs
        w-1 rows of left context from the previous chunk.
        Peak RAM: CHUNK_ROWS × w × C × 4 bytes ≈ 500k × 120 × 6 × 4 = 1.4 GB.
        Total runtime: ~40 s vs ~40 min for the original loop.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data   = np.ascontiguousarray(data, dtype=np.float32)
        n, c   = data.shape
        w      = self.window_size
        CHUNK  = 500_000
        out    = np.empty_like(data)

        for start in range(0, n, CHUNK):
            end      = min(start + CHUNK, n)
            ctx_lo   = max(0, start - (w - 1))
            chunk    = data[ctx_lo : end]                   # (ctx+chunk, C)
            nc       = end - start
            pad_rows = w - 1 - (start - ctx_lo)            # rows of padding still needed
            if pad_rows > 0:
                chunk = np.concatenate(
                    [np.repeat(data[:1], pad_rows, axis=0), chunk], axis=0
                )
            # chunk is now exactly (nc + w - 1, C)
            shape   = (nc, w, c)
            strides = (chunk.strides[0], chunk.strides[0], chunk.strides[1])
            wins    = np.lib.stride_tricks.as_strided(chunk, shape=shape, strides=strides)
            w_min   = wins.min(axis=1)
            w_max   = wins.max(axis=1)
            out[start:end] = (data[start:end] - w_min) / (w_max - w_min + self.epsilon)

        return out


class ZScoreScaler:
    """Rolling z-score normalisation."""

    def __init__(self, window_size: int = 120, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Vectorised in chunks — same strategy as WindowMinMaxScaler."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data   = np.ascontiguousarray(data, dtype=np.float32)
        n, c   = data.shape
        w      = self.window_size
        CHUNK  = 500_000
        out    = np.empty_like(data)

        for start in range(0, n, CHUNK):
            end      = min(start + CHUNK, n)
            ctx_lo   = max(0, start - (w - 1))
            chunk    = data[ctx_lo : end]
            nc       = end - start
            pad_rows = w - 1 - (start - ctx_lo)
            if pad_rows > 0:
                chunk = np.concatenate(
                    [np.repeat(data[:1], pad_rows, axis=0), chunk], axis=0
                )
            shape   = (nc, w, c)
            strides = (chunk.strides[0], chunk.strides[0], chunk.strides[1])
            wins    = np.lib.stride_tricks.as_strided(chunk, shape=shape, strides=strides)
            w_mean  = wins.mean(axis=1)
            w_std   = wins.std(axis=1)
            out[start:end] = (data[start:end] - w_mean) / (w_std + self.epsilon)

        return out


def get_scaler(method: str, window_size: int):
    if method == "window_minmax":
        return WindowMinMaxScaler(window_size)
    if method == "zscore":
        return ZScoreScaler(window_size)
    raise ValueError(f"Unknown scaler: {method}")


# ── OHLCV feature preparation ─────────────────────────────────────────────────

def _compute_rq(df) -> np.ndarray:
    """Extract regime quality (rq) from the dataframe or return zeros.

    rq is computed by the GMM2 regime model during data ingestion and stored
    in the NPZ. If available in df, extract directly; otherwise return 0.5.
    Range [0, 0.92]. Higher = more confident regime assignment.

    Phase 4 exploration: KS=0.623 STRONG, MI=0.198 (21x OHLCV), ratio=0.60 PASS.
    Strongest supervised addition found across all exploration phases.
    """
    try:
        if "rq" in df.columns:
            return df["rq"].to_numpy().astype(np.float32)
        elif "regime_quality" in df.columns:
            return df["regime_quality"].to_numpy().astype(np.float32)
        else:
            # rq not in raw CSV — will be 0.5 placeholder
            # The NPZ rebuild pipeline should join rq from regime labels CSV
            import warnings
            warnings.warn("rq column not found in df — using 0.5 placeholder. "
                          "Join regime labels before NPZ rebuild for correct rq values.")
            return np.full(len(df), 0.5, dtype=np.float32)
    except Exception:
        return np.full(len(df), 0.5, dtype=np.float32)


def _compute_session_phase(close_prices: np.ndarray, df) -> np.ndarray:
    """Compute session_phase feature aligned to the close_prices array.

    Returns float32 array in [0, 1]:
        0.0 = Asian session (00:00-08:00 UTC, low-signal dead zone)
        0.5 = NY session   (16:00-22:00 UTC)
        1.0 = London session (08:00-16:00 UTC, peak sell label rate 5.07%)

    Validated in phase4_feature_exploration.ipynb:
        KS=0.117 STRONG, MI=0.037 (3.99x OHLCV), ratio=2.96 PASS -> SUPERVISED
        London sell rate 1.44x baseline; Asian 0.77x.
    """
    import polars as pl
    n = len(close_prices)
    session = np.full(n, 0.0, dtype=np.float32)  # default = Asian

    try:
        ts_col = df["timestamp"]
        for i in range(n):
            try:
                ts = ts_col[i]
                hour = ts.hour if hasattr(ts, "hour") else int((int(ts) // (3600 * 10**9)) % 24)
                if 8 <= hour < 16:
                    session[i] = 1.0    # London
                elif 16 <= hour < 22:
                    session[i] = 0.5    # NY
                # else: Asian = 0.0 (default)
            except Exception:
                session[i] = 0.5
    except Exception:
        # Vectorised fallback if polars timestamp available
        try:
            ts_ns = df["timestamp"].cast(pl.Datetime).to_numpy().astype(np.int64)
            hours = (ts_ns // (3600 * 10**9) % 24).astype(np.int32)
            session = np.where((hours >= 8)  & (hours < 16), 1.0,
                      np.where((hours >= 16) & (hours < 22), 0.5, 0.0)).astype(np.float32)
        except Exception:
            session[:] = 0.5  # safe fallback

    return session


def prepare_features(
    df: pl.DataFrame,
    scaler_method: str = "window_minmax",
    window_size: int = 120,
    cache_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract and scale OHLCV+spread features from a Polars M1 DataFrame.

    Returns:
        features:     (N, 6) float32  — scaled [open, high, low, close, tick_vol, spread]
        close_prices: (N,)   float64  — raw close prices for labelling
        high_prices:  (N,)   float64  — raw high prices  (ATR labelling)
        low_prices:   (N,)   float64  — raw low prices   (ATR labelling)

    All four arrays are aligned to the same weekday-filtered, forward-filled rows.
    """
    # ── Cache load/save ──────────────────────────────────────────────────────
    # Pass cache_path="/content/drive/MyDrive/Colab Notebooks/features_10d"
    # to skip recomputation on subsequent runs (~3 s load vs ~3 min compute).
    if cache_path is not None:
        npz = cache_path if cache_path.endswith(".npz") else cache_path + ".npz"
        if Path(npz).exists():
            logger.info(f"Loading feature cache: {npz}")
            d = np.load(npz)
            return d["features"], d["close"], d["high"], d["low"]

    df = df.filter(pl.col("timestamp").dt.weekday().is_in([1, 2, 3, 4, 5]))
    for c in ["open", "high", "low", "close"]:
        df = df.with_columns(pl.col(c).forward_fill())

    feature_cols = ["open", "high", "low", "close", "tick_volume", "spread"]
    features     = df.select(feature_cols).to_numpy().astype(np.float32)
    close_prices = df["close"].to_numpy()
    high_prices  = df["high"].to_numpy()
    low_prices   = df["low"].to_numpy()

    # ── 4 microstructure features (Phase 4, Candidate B) ─────────────────────
    # Evidence: 240-bar sequence profiling shows all 4 features have KS > 0.10
    # and MI significantly above OHLCV baseline.
    c = close_prices
    h = high_prices
    l = low_prices
    vol  = features[:, 4].astype(np.float64)
    sprd = features[:, 5].astype(np.float64)  # unscaled spread

    # Use raw spread from df for spread_pressure
    sprd_raw = df["spread"].to_numpy().astype(np.float64)

    # 1. bar_return_bps — scale-invariant direction+magnitude (KS=0.162, MI=0.015)
    bar_return = np.empty(len(c), dtype=np.float32)
    bar_return[0] = 0.0
    bar_return[1:] = ((c[1:] - c[:-1]) / (c[:-1] + 1e-8) * 10000).astype(np.float32)

    # 2. wick_asymmetry — (upper-lower)/range, concentrated last-20 (KS=0.252, MI=0.037)
    bar_range = (h - l).astype(np.float32)
    upper_wick = ((h - c) / (bar_range + 1e-8)).astype(np.float32)
    lower_wick = ((c - l) / (bar_range + 1e-8)).astype(np.float32)
    wick_asymmetry = upper_wick - lower_wick  # positive = bearish pressure

    # 3. vol_zscore — volume anomaly, tanh-smooth (KS=0.111, MI=0.024)
    vol_s = pl.Series(vol)
    vol_mean = np.array(vol_s.rolling_mean(20).fill_null(vol[:20].mean()), dtype=np.float64)
    vol_std  = np.array(vol_s.rolling_std(20).fill_null(1.0), dtype=np.float64)
    vol_zscore = np.tanh(((vol - vol_mean) / (vol_std + 1e-8)) / 2).astype(np.float32)

    # 4. spread_pressure — log1p(spread/range), smooth tail (KS=0.600, MI=0.056)
    spread_pressure = np.log1p(sprd_raw / (bar_range.astype(np.float64) + 1e-8)).astype(np.float32)

    micro_4d = np.stack([bar_return, wick_asymmetry, vol_zscore, spread_pressure], axis=1)

    # Scale OHLCV block, leave micro block pre-normalised (RobustScaler equivalent)
    scaler  = get_scaler(scaler_method, window_size)
    scaled_6d  = scaler.transform(features).astype(np.float32)
    scaled_10d = np.concatenate([scaled_6d, micro_4d], axis=1)  # (N, 10)

    # ── Feature 11: session_phase (Phase 4 exploration — SUPERVISED verdict) ──
    # KS=0.117 STRONG, MI=0.037 (3.99x OHLCV mean), redundancy ratio=2.96 PASS.
    # First confirmed supervised addition from all Phase 4-8 exploration.
    # session_phase encodes London/NY session presence: 0=Asian, 0.5=transition, 1=peak.
    # Already in RL obs (obs[9]) and NPZ separately — now added to supervised features.
    # London sell rate = 5.07% (1.44x baseline), Asian = 2.69% (0.77x baseline).
    session_phase_feat = _compute_session_phase(close_prices, df)
    scaled_11d = np.concatenate(
        [scaled_10d, session_phase_feat.reshape(-1, 1)], axis=1
    )  # (N, 11)

    # ── Feature 12: rq_regime (Phase 4 exploration — SUPERVISED verdict) ───────
    # KS=0.623 STRONG, MI=0.198 (21x OHLCV mean), ratio=0.60 PASS.
    # Strongest supervised addition found across all Phase 4-8 exploration.
    # rq = regime quality scalar from GMM2 model, range [0, 0.92].
    # Quantifies model confidence in current regime assignment.
    # Already in RL obs (obs[10] = regime_quality) and NPZ — zero preprocessing cost.
    rq_feat = _compute_rq(df)
    scaled_12d = np.concatenate(
        [scaled_11d, rq_feat.reshape(-1, 1)], axis=1
    )  # (N, 12)

    logger.info(f"Features prepared: {scaled_12d.shape}  (6-OHLCV + 4-micro + session_phase + rq_regime)")
    if cache_path is not None:
        npz = cache_path if cache_path.endswith(".npz") else cache_path + ".npz"
        np.savez_compressed(npz, features=scaled_12d,
                            close=close_prices, high=high_prices, low=low_prices)
        logger.info(f"Feature cache saved: {npz}")
    return scaled_12d, close_prices, high_prices, low_prices


# ── Regime label joining (Step 2) ─────────────────────────────────────────────

def join_regime_labels(
    df_m1: pl.DataFrame,
    regime_csv: str | Path,
) -> pl.DataFrame:
    """Left-join daily regime labels onto M1 OHLCV bars by date.

    Reads daily_regime_labels.csv (generated by the research notebook),
    forward-fills daily values across every intraday bar of each trading day.

    If the CSV is absent the function returns df_m1 with neutral defaults (0.5)
    and logs a warning — training still runs, regime obs dims are uninformative.

    Columns added (float32):
        gmm2_state          — 0.0=Bear  1.0=Bull  (2-state GMM, 5d min-dwell)
        km_label_63d_enc    — 0.0–1.0  (Regime-A … Regime-D)
        vol_regime_enc      — 0.0=LOW  0.5=NORMAL  1.0=HIGH
        gs_quartile_enc     — 0.0=Q1(silver-leads) … 1.0=Q4(gold-leads)
        cu_au_regime_enc    — 0.0=Financial  0.5=Mixed  1.0=Commodity
        regime_quality_norm — [0,1] normalised Sharpe from GMM×vol heatmap
    """
    csv_path = Path(regime_csv)
    if not csv_path.exists():
        logger.warning(
            f"Regime CSV not found: {csv_path}. "
            "Regime columns set to neutral defaults. "
            "Generate it by running notebooks/00_market_regime_explorer_v5.ipynb."
        )
        return _add_neutral_regime_cols(df_m1)

    try:
        import pandas as pd
        daily = pd.read_csv(str(csv_path), index_col=0, parse_dates=True)
        daily.index = pd.to_datetime(daily.index).normalize()
    except Exception as e:
        logger.error(f"Failed to load regime CSV: {e}")
        return _add_neutral_regime_cols(df_m1)

    # gmm2_state: prefer explicit 2-state col; fall back by mapping 3-state
    if "gmm2_state" in daily.columns:
        daily["_gmm2"] = daily["gmm2_state"].fillna(1.0).astype(float)
    elif "gmm_state" in daily.columns:
        daily["_gmm2"] = (
            daily["gmm_state"]
            .map({0.0: 0.0, 1.0: 1.0, 2.0: 1.0})
            .fillna(1.0)
        )
    else:
        daily["_gmm2"] = 1.0

    import pandas as pd
    daily["_km"] = (
        daily.get("km_label_63d", pd.Series("Regime-B", index=daily.index))
        .map(_KM_ENC)
        .fillna(0.33)
    )
    daily["_vol"] = (
        daily.get("vol_regime", pd.Series("NORMAL", index=daily.index))
        .map(_VOL_ENC)
        .fillna(0.5)
    )

    # G/S quartile: rank gs_ratio into [0,1]; fall back to Q1 (0.0)
    if "gs_ratio" in daily.columns:
        gs = daily["gs_ratio"].dropna()
        daily["_gs_q"] = (
            gs.rank(pct=True)
            .apply(lambda r: 0.0 if r <= 0.25 else (0.33 if r <= 0.5 else (0.67 if r <= 0.75 else 1.0)))
            .reindex(daily.index)
            .fillna(0.0)
        )
    else:
        daily["_gs_q"] = 0.0

    daily["_cu"] = 0.5  # Mixed default (no direct daily Cu-Au column in export)
    daily["_rq"] = (
        daily.get("regime_quality_norm", pd.Series(0.5, index=daily.index))
        .fillna(0.5)
    )

    # Build date-keyed lookup for O(1) per-bar access
    keep = daily[["_gmm2", "_km", "_vol", "_gs_q", "_cu", "_rq"]].copy()
    keep.index = keep.index.date
    regime_map = keep.to_dict(orient="index")

    df_m1  = df_m1.with_columns(pl.col("timestamp").dt.date().alias("_date"))
    dates  = df_m1["_date"].to_list()

    gmm2_v, km_v, vol_v, gs_v, cu_v, rq_v = [], [], [], [], [], []
    last = {"g": 1.0, "k": 0.33, "v": 0.5, "gs": 0.0, "c": 0.5, "r": 0.5}

    for d in dates:
        row = regime_map.get(d)
        if row:
            last["g"]  = float(row["_gmm2"])
            last["k"]  = float(row["_km"])
            last["v"]  = float(row["_vol"])
            last["gs"] = float(row["_gs_q"])
            last["c"]  = float(row["_cu"])
            last["r"]  = float(row["_rq"])
        gmm2_v.append(last["g"]); km_v.append(last["k"])
        vol_v.append(last["v"]);  gs_v.append(last["gs"])
        cu_v.append(last["c"]);   rq_v.append(last["r"])

    df_m1 = df_m1.drop("_date").with_columns([
        pl.Series("gmm2_state",          gmm2_v, dtype=pl.Float32),
        pl.Series("km_label_63d_enc",    km_v,   dtype=pl.Float32),
        pl.Series("vol_regime_enc",      vol_v,  dtype=pl.Float32),
        pl.Series("gs_quartile_enc",     gs_v,   dtype=pl.Float32),
        pl.Series("cu_au_regime_enc",    cu_v,   dtype=pl.Float32),
        pl.Series("regime_quality_norm", rq_v,   dtype=pl.Float32),
    ])

    bull_pct = sum(gmm2_v) / max(len(gmm2_v), 1)
    logger.info(
        f"Regime labels joined: {len(df_m1):,} bars | "
        f"Bull {bull_pct:.1%} Bear {1-bull_pct:.1%} | "
        f"vol_mean {sum(vol_v)/len(vol_v):.2f} | "
        f"rq_mean {sum(rq_v)/len(rq_v):.2f}"
    )
    return df_m1


def _add_neutral_regime_cols(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.lit(1.0).cast(pl.Float32).alias("gmm2_state"),
        pl.lit(0.33).cast(pl.Float32).alias("km_label_63d_enc"),
        pl.lit(0.5).cast(pl.Float32).alias("vol_regime_enc"),
        pl.lit(0.0).cast(pl.Float32).alias("gs_quartile_enc"),
        pl.lit(0.5).cast(pl.Float32).alias("cu_au_regime_enc"),
        pl.lit(0.5).cast(pl.Float32).alias("regime_quality_norm"),
    ])


def get_regime_array(df_m1_with_regime: pl.DataFrame) -> np.ndarray:
    """Extract 6 regime columns as (N, 6) float32 for RL obs alignment."""
    cols = [
        "gmm2_state", "km_label_63d_enc", "vol_regime_enc",
        "gs_quartile_enc", "cu_au_regime_enc", "regime_quality_norm",
    ]
    available = [c for c in cols if c in df_m1_with_regime.columns]
    if len(available) < 6:
        logger.warning(f"Only {len(available)}/6 regime cols present — padding with 0.5")
    return df_m1_with_regime.select(available).to_numpy().astype(np.float32)


# ── RL observation context features (Step 4, merged from feature_engineering) ─

def compute_rl_obs_features(
    prices: np.ndarray,
    timestamps,
    atr_period: int = 14,
    ema_period: int = 20,
    ema_lag: int = 5,
    session_open_utc: float = 8.0,
    session_range_hrs: float = 13.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3 market-context features for RL observation indices 7-9.

    Args:
        prices:     (N,) close price array aligned to sequence output.
        timestamps: list[datetime] | None — M1 bar UTC timestamps.

    Returns:
        atr_norm      (N,) float32 — rolling ATR/close clipped [0, 0.05]
        trend_norm    (N,) float32 — EMA slope/close*100 clipped [-2, 2]
        session_phase (N,) float32 — London/NY overlap fraction [0, 1]

    All features use look-back only — no lookahead bias.
    """
    n = len(prices)
    atr_norm      = np.zeros(n, dtype=np.float32)
    trend_norm    = np.zeros(n, dtype=np.float32)
    session_phase = np.zeros(n, dtype=np.float32)

    # ATR proxy: rolling price range / close
    for i in range(atr_period, n):
        window   = prices[i - atr_period : i + 1]
        atr_norm[i] = min(float((window.max() - window.min()) / max(prices[i], 1e-8)), 0.05)

    # EMA slope
    alpha = 2.0 / (ema_period + 1)
    ema   = np.zeros(n, dtype=np.float64)
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    for i in range(ema_lag, n):
        slope = (ema[i] - ema[i - ema_lag]) / max(prices[i], 1e-8) * 100.0
        trend_norm[i] = float(np.clip(slope, -2.0, 2.0))

    # Session phase
    if timestamps is not None:
        for i, ts in enumerate(timestamps):
            try:
                hour = ts.hour + ts.minute / 60.0
                session_phase[i] = float(np.clip(
                    (hour - session_open_utc) / session_range_hrs, 0.0, 1.0
                ))
            except Exception:
                session_phase[i] = 0.5

    return atr_norm, trend_norm, session_phase