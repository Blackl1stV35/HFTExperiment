"""Data preprocessing: scaling, feature extraction, and regime label joining.

Phase 3 additions:
    join_regime_labels() — left-joins daily_regime_labels.csv onto M1 bars
    by date, forward-filling across intraday bars. Adds 6 columns that the
    RL observation vector (obs 10→13) and supervised sampler consume.

    Columns added per M1 bar:
        gmm2_state          int   0=Bear 1=Bull (2-state GMM, 5d min-dwell)
        km_label_63d_enc    float 0.0–1.0 (Regime-A=0.0 … Regime-D=1.0)
        vol_regime_enc      float 0.0=LOW 0.5=NORMAL 1.0=HIGH
        gs_quartile_enc     float 0.0=Q1(silver-leads) … 1.0=Q4(gold-leads)
        cu_au_regime_enc    float 0.0=Financial 0.5=Mixed 1.0=Commodity
        regime_quality_norm float [0,1] normalised Sharpe from 2D heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger


# ── Scalers ───────────────────────────────────────────────────────────────────

class WindowMinMaxScaler:
    def __init__(self, window_size: int = 120, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon

    def transform(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n, f = data.shape
        scaled = np.zeros_like(data)
        for i in range(n):
            start = max(0, i - self.window_size + 1)
            window = data[start : i + 1]
            w_min = window.min(axis=0)
            w_max = window.max(axis=0)
            scaled[i] = (data[i] - w_min) / (w_max - w_min + self.epsilon)
        return scaled


class ZScoreScaler:
    def __init__(self, window_size: int = 120, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon

    def transform(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n, f = data.shape
        scaled = np.zeros_like(data)
        for i in range(n):
            start = max(0, i - self.window_size + 1)
            window = data[start : i + 1]
            scaled[i] = (data[i] - window.mean(axis=0)) / (window.std(axis=0) + self.epsilon)
        return scaled


def get_scaler(method: str, window_size: int):
    if method == "window_minmax":
        return WindowMinMaxScaler(window_size)
    elif method == "zscore":
        return ZScoreScaler(window_size)
    raise ValueError(f"Unknown scaler: {method}")


# ── Core feature preparation ───────────────────────────────────────────────────

def prepare_features(
    df: pl.DataFrame,
    scaler_method: str = "window_minmax",
    window_size: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and scale OHLCV+spread features.

    Returns:
        features: (n_samples, 6) scaled OHLCV+spread
        close_prices: (n_samples,) raw close prices
    """
    df = df.filter(pl.col("timestamp").dt.weekday().is_in([1, 2, 3, 4, 5]))
    for c in ["open", "high", "low", "close"]:
        df = df.with_columns(pl.col(c).forward_fill())

    feature_cols = ["open", "high", "low", "close", "tick_volume", "spread"]
    features = df.select(feature_cols).to_numpy().astype(np.float32)
    close_prices = df["close"].to_numpy()

    scaler = get_scaler(scaler_method, window_size)
    features_scaled = scaler.transform(features).astype(np.float32)
    logger.info(f"Features prepared: {features_scaled.shape}")
    return features_scaled, close_prices


# ── Step 2: Regime label joining ───────────────────────────────────────────────

# Encoding maps — must match notebook export values
_KM_ENC = {
    "Regime-A (best)":  0.0,
    "Regime-B":         0.33,
    "Regime-C":         0.67,
    "Regime-D (worst)": 1.0,
}
_VOL_ENC = {"LOW": 0.0, "NORMAL": 0.5, "HIGH": 1.0}
_CU_ENC  = {"Financial": 0.0, "Mixed": 0.5, "Commodity": 1.0}


def join_regime_labels(
    df_m1: pl.DataFrame,
    regime_csv: str | Path,
) -> pl.DataFrame:
    """Left-join daily regime labels onto M1 OHLCV bars.

    Reads ``daily_regime_labels.csv`` (produced by the research notebook),
    extracts date from M1 timestamps, and forward-fills daily values across
    all intraday bars of each trading day.

    Args:
        df_m1:      Polars DataFrame with a ``timestamp`` column (M1 bars).
        regime_csv: Path to ``data/regime/daily_regime_labels.csv``.

    Returns:
        df_m1 with 6 new float32 columns:
            gmm2_state, km_label_63d_enc, vol_regime_enc,
            gs_quartile_enc, cu_au_regime_enc, regime_quality_norm
    """
    csv_path = Path(regime_csv)
    if not csv_path.exists():
        logger.warning(
            f"Regime CSV not found at {csv_path}. "
            "All regime columns will be set to 0.5 (neutral). "
            "Run the research notebook first to generate daily_regime_labels.csv."
        )
        n = len(df_m1)
        return df_m1.with_columns([
            pl.lit(1.0).cast(pl.Float32).alias("gmm2_state"),
            pl.lit(0.33).cast(pl.Float32).alias("km_label_63d_enc"),
            pl.lit(0.5).cast(pl.Float32).alias("vol_regime_enc"),
            pl.lit(0.0).cast(pl.Float32).alias("gs_quartile_enc"),
            pl.lit(0.5).cast(pl.Float32).alias("cu_au_regime_enc"),
            pl.lit(0.5).cast(pl.Float32).alias("regime_quality_norm"),
        ])

    try:
        import pandas as pd
        daily = pd.read_csv(str(csv_path), index_col=0, parse_dates=True)
        daily.index = pd.to_datetime(daily.index).normalize()  # date only
    except Exception as e:
        logger.error(f"Failed to load regime CSV: {e}")
        return _add_neutral_regime_cols(df_m1)

    # ── Encode categorical columns ─────────────────────────────────────────────
    # gmm2_state: prefer gmm2_state column; fall back to gmm_state mapping
    if "gmm2_state" in daily.columns:
        daily["_gmm2"] = daily["gmm2_state"].fillna(1.0).astype(float)
    elif "gmm_state" in daily.columns:
        # Map 3-state (0=Bear,1=Neutral,2=Bull) → 2-state (0=Bear,1=Bull)
        daily["_gmm2"] = daily["gmm_state"].map({0.0: 0.0, 1.0: 1.0, 2.0: 1.0}).fillna(1.0)
    else:
        daily["_gmm2"] = 1.0  # default Bull (no-restriction)

    daily["_km"] = daily.get("km_label_63d", pd.Series("Regime-B", index=daily.index)).map(
        _KM_ENC).fillna(0.33)

    daily["_vol"] = daily.get("vol_regime", pd.Series("NORMAL", index=daily.index)).map(
        _VOL_ENC).fillna(0.5)

    # G/S quartile: derive from gs_ratio rank or use default Q1 (0.0)
    if "gs_ratio" in daily.columns:
        gs = daily["gs_ratio"].dropna()
        daily["_gs_q"] = gs.rank(pct=True).apply(
            lambda r: 0.0 if r <= 0.25 else (0.33 if r <= 0.5 else (0.67 if r <= 0.75 else 1.0))
        ).reindex(daily.index).fillna(0.0)
    else:
        daily["_gs_q"] = 0.0

    # Cu-Au regime: derive from xau_vol_ann_pct as a proxy if not present
    # (The notebook exports vol_regime which captures the same info)
    daily["_cu"] = 0.5  # default Mixed

    daily["_rq"] = daily.get("regime_quality_norm", pd.Series(0.5, index=daily.index)).fillna(0.5)

    # Select and resample to fill weekends/gaps
    keep = daily[["_gmm2","_km","_vol","_gs_q","_cu","_rq"]].copy()

    # ── Join to M1 timestamps ──────────────────────────────────────────────────
    # Extract date from M1 DataFrame
    df_m1 = df_m1.with_columns(
        pl.col("timestamp").dt.date().alias("_date")
    )
    dates = df_m1["_date"].to_list()

    # Build lookup dict: date → regime row
    keep.index = keep.index.date  # drop time component
    regime_map = keep.to_dict(orient="index")

    # Map each M1 bar's date to regime values; forward-fill missing dates
    gmm2_vals = []
    km_vals   = []
    vol_vals  = []
    gs_vals   = []
    cu_vals   = []
    rq_vals   = []

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
        gmm2_vals.append(last["g"])
        km_vals.append(last["k"])
        vol_vals.append(last["v"])
        gs_vals.append(last["gs"])
        cu_vals.append(last["c"])
        rq_vals.append(last["r"])

    df_m1 = df_m1.drop("_date").with_columns([
        pl.Series("gmm2_state",       gmm2_vals, dtype=pl.Float32),
        pl.Series("km_label_63d_enc", km_vals,   dtype=pl.Float32),
        pl.Series("vol_regime_enc",   vol_vals,  dtype=pl.Float32),
        pl.Series("gs_quartile_enc",  gs_vals,   dtype=pl.Float32),
        pl.Series("cu_au_regime_enc", cu_vals,   dtype=pl.Float32),
        pl.Series("regime_quality_norm", rq_vals, dtype=pl.Float32),
    ])

    bull_pct = sum(gmm2_vals) / len(gmm2_vals)
    logger.info(
        f"Regime labels joined: {len(df_m1):,} M1 bars | "
        f"Bull {bull_pct:.1%} | "
        f"Vol enc mean {sum(vol_vals)/len(vol_vals):.2f} | "
        f"Regime quality mean {sum(rq_vals)/len(rq_vals):.2f}"
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
    """Extract 6 regime columns as (N, 6) float32 array for RL obs concat."""
    cols = [
        "gmm2_state", "km_label_63d_enc", "vol_regime_enc",
        "gs_quartile_enc", "cu_au_regime_enc", "regime_quality_norm",
    ]
    available = [c for c in cols if c in df_m1_with_regime.columns]
    if len(available) < 6:
        logger.warning(f"Only {len(available)}/6 regime cols present — padding with 0.5")
    arr = df_m1_with_regime.select(available).to_numpy().astype(np.float32)
    return arr
