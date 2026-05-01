#!/usr/bin/env python3
"""Precompute and save all training-ready arrays into a single .npz file.

Run this LOCALLY before uploading to Colab. Output is one file that
train_supervised.py and train_rl.py load in ~8 seconds, replacing:
    - DuckDB SQL deserialisation     (~4 min on Colab)
    - join_regime_labels Python loop (~3 min on Colab)
    - prepare_features chunked scaler (~40 s on Colab)
    - ATR labelling                   (~5 s vectorised / was 25 min loop)
    - compute_rl_obs_features loops   (~2 min on Colab)

Total Colab startup saved: ~10 min → ~8 s per run.

Output file layout (all arrays aligned, window_size already trimmed):
    features        (N, 10)  float32  — scaled OHLCV + 4 micro features
    labels          (N,)     int64    — ATR-adaptive labels (sell=0,hold=1,buy=2)
    close           (N,)     float64  — raw close prices
    high            (N,)     float64  — raw high prices
    low             (N,)     float64  — raw low prices
    timestamps_ns   (N,)     int64    — UTC nanoseconds (for session_phase)
    gmm2            (N,)     float32  — regime: 0=Bear 1=Bull
    km_enc          (N,)     float32  — regime: km_label_63d encoded
    vol_enc         (N,)     float32  — regime: LOW=0 NORMAL=0.5 HIGH=1
    gs_q            (N,)     float32  — G/S quartile [0,1]
    cu_au           (N,)     float32  — Cu-Au regime encoded
    rq              (N,)     float32  — regime quality normalised [0,1]
    atr_norm        (N,)     float32  — RL obs feature 7
    trend_norm      (N,)     float32  — RL obs feature 8
    session_phase   (N,)     float32  — RL obs feature 9
    metadata        scalar   — JSON string with config snapshot

N = len(bars) - window_size  (sequences aligned with seq_len context)

Usage:
    python scripts/precompute_features.py \\
        --data-dir data \\
        --symbol XAUUSD \\
        --regime-csv data/regime/daily_regime_labels.csv \\
        --window-size 240 \\
        --output data/training_ready.npz

    # Then upload training_ready.npz to Google Drive and set in config:
    # paths.training_ready: /content/drive/MyDrive/Colab Notebooks/training_ready.npz
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from src.data.preprocessing import (
    prepare_features,
    join_regime_labels,
    get_regime_array,
    compute_rl_obs_features,
)
from src.data.tick_store import TickStore
from src.training.labels import LabelConfig, get_labeler, ATRAdaptiveLabeler


def main():
    parser = argparse.ArgumentParser(description="Precompute training-ready .npz")
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--symbol",      default="XAUUSD")
    parser.add_argument("--timeframe",   default="M1")
    parser.add_argument("--regime-csv",  default="data/regime/daily_regime_labels.csv")
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--scaler",      default="window_minmax")
    parser.add_argument("--atr-period",  type=int, default=14)
    parser.add_argument("--atr-mult-tp", type=float, default=1.5)
    parser.add_argument("--atr-mult-sl", type=float, default=0.75)   # failed: sell:buy ratio = 4.60×  (was 4.76×, target ≈ 1.0-2.0×)
    parser.add_argument("--atr-min-tp",  type=float, default=150.0)
    parser.add_argument("--atr-max-tp",  type=float, default=800.0)
    parser.add_argument("--max-holding", type=int, default=40)
    parser.add_argument("--pip-value",   type=float, default=0.10)
    parser.add_argument("--output",      default="data/training_ready.npz")
    args = parser.parse_args()

    t_total = time.time()

    # ── 1. Load raw M1 data ──────────────────────────────────────────────────
    logger.info(f"Loading {args.symbol} {args.timeframe} from {args.data_dir}/ticks.duckdb")
    t0 = time.time()
    store = TickStore(f"{args.data_dir}/ticks.duckdb")
    df    = store.query_ohlcv(args.symbol, args.timeframe)
    store.close()
    logger.info(f"  {len(df):,} bars loaded in {time.time()-t0:.1f}s")

    # ── 2. Join regime labels ────────────────────────────────────────────────
    logger.info("Joining regime labels...")
    t0 = time.time()
    df = join_regime_labels(df, args.regime_csv)
    logger.info(f"  Regime join done in {time.time()-t0:.1f}s")

    # ── 3. Prepare 10-dim features (scaler + microstructure) ─────────────────
    logger.info("Computing features (chunked scaler + microstructure)...")
    t0 = time.time()
    features, close, high, low = prepare_features(
        df, scaler_method=args.scaler, window_size=args.window_size
    )
    logger.info(f"  Features {features.shape} done in {time.time()-t0:.1f}s")

    ws = args.window_size

    # ── 4. ATR-adaptive labels ───────────────────────────────────────────────
    logger.info("Computing ATR-adaptive labels (vectorised)...")
    t0 = time.time()
    label_cfg = LabelConfig(
        method             = "atr_adaptive",
        atr_period         = args.atr_period,
        atr_multiplier_tp  = args.atr_mult_tp,
        atr_multiplier_sl  = args.atr_mult_sl,
        atr_min_tp_pips    = args.atr_min_tp,
        atr_max_tp_pips    = args.atr_max_tp,
        max_holding_bars   = args.max_holding,
        pip_value          = args.pip_value,
    )
    labeler = get_labeler(label_cfg)
    labels_full = (labeler.label(close, high, low)
                   if isinstance(labeler, ATRAdaptiveLabeler)
                   else labeler.label(close))
    logger.info(f"  Labels {labels_full.shape} done in {time.time()-t0:.1f}s")

    # Trim warmup window — all arrays now start at index ws
    features = features[ws:]
    labels   = labels_full[ws:]
    close    = close[ws:]
    high     = high[ws:]
    low      = low[ws:]

    n = len(features)
    logger.info(f"  After trim: {n:,} sequences | "
                f"sell={np.sum(labels==0):,} hold={np.sum(labels==1):,} "
                f"buy={np.sum(labels==2):,}")

    # ── 5. Build weekday-filtered df once — used for BOTH regime + timestamps ─
    # prepare_features filters weekdays internally → 5,680,771 bars.
    # df (raw) has 5,998,591 bars including weekends.
    # get_regime_array and timestamp extraction must use the same filtered view.
    import polars as pl
    import pandas as pd
    df_wd = df.filter(pl.col("timestamp").dt.weekday().is_in([1, 2, 3, 4, 5]))
    assert len(df_wd) == len(features) + ws,         f"Weekday df length {len(df_wd)} != features {len(features)+ws}"

    # ── 6. Regime arrays (from weekday-filtered df) ──────────────────────────
    logger.info("Extracting regime arrays...")
    t0 = time.time()
    regime_full = get_regime_array(df_wd)            # (5680771, 6)
    regime      = regime_full[ws:]                   # (n, 6)
    assert len(regime) == n, f"Regime length {len(regime)} != n {n}"
    gmm2    = regime[:, 0]
    km_enc  = regime[:, 1]
    vol_enc = regime[:, 2]
    gs_q    = regime[:, 3]
    cu_au   = regime[:, 4]
    rq      = regime[:, 5]
    logger.info(f"  Bull={gmm2.mean():.1%}  Bear={1-gmm2.mean():.1%}  "
                f"done in {time.time()-t0:.1f}s")

    # ── 7. Timestamps (from df_wd already built above) ─────────────────────
    logger.info("Extracting timestamps (weekday-filtered)...")
    t0 = time.time()
    if "timestamp" in df_wd.columns:
        ts_list = df_wd["timestamp"].to_list()[ws:]
        if len(ts_list) != n:
            raise ValueError(
                f"Timestamp/features mismatch: {len(ts_list)} vs {n} — "
                "check weekday filter consistency"
            )
        timestamps_ns = np.array(
            [int(pd.Timestamp(t).value) for t in ts_list], dtype=np.int64
        )
    else:
        timestamps_ns = np.zeros(n, dtype=np.int64)
        logger.warning("No timestamp column — session_phase will be 0.5")
    logger.info(f"  Timestamps {len(timestamps_ns):,} done in {time.time()-t0:.1f}s")

    # ── 8. RL observation context features (atr_norm, trend, session) ────────
    logger.info("Computing RL obs context features...")
    t0 = time.time()
    if timestamps_ns.any():
        ts_dt = pd.to_datetime(timestamps_ns).to_pydatetime().tolist()
    else:
        ts_dt = None
    atr_norm, trend_norm, session_phase = compute_rl_obs_features(close, ts_dt)
    logger.info(f"  RL features done in {time.time()-t0:.1f}s")

    # ── 9. Save single .npz ──────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = json.dumps({
        "symbol":       args.symbol,
        "timeframe":    args.timeframe,
        "window_size":  args.window_size,
        "scaler":       args.scaler,
        "n_sequences":  n,
        "label_method": "atr_adaptive",
        "atr_mult_tp":  args.atr_mult_tp,
        "atr_mult_sl":  args.atr_mult_sl,
        "atr_min_tp":   args.atr_min_tp,
        "atr_max_tp":   args.atr_max_tp,
        "max_holding":  args.max_holding,
        "feature_dim":  features.shape[1],
        "created":      time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    })

    logger.info(f"Saving to {out_path} ...")
    t0 = time.time()
    np.savez_compressed(
        str(out_path),
        features      = features.astype(np.float32),
        labels        = labels.astype(np.int64),
        close         = close.astype(np.float64),
        high          = high.astype(np.float64),
        low           = low.astype(np.float64),
        timestamps_ns = timestamps_ns,
        gmm2          = gmm2.astype(np.float32),
        km_enc        = km_enc.astype(np.float32),
        vol_enc       = vol_enc.astype(np.float32),
        gs_q          = gs_q.astype(np.float32),
        cu_au         = cu_au.astype(np.float32),
        rq            = rq.astype(np.float32),
        atr_norm      = atr_norm.astype(np.float32),
        trend_norm    = trend_norm.astype(np.float32),
        session_phase = session_phase.astype(np.float32),
        metadata      = np.array(metadata),
    )

    size_mb = out_path.stat().st_size / 1024**2
    logger.info(f"  Saved {size_mb:.0f} MB in {time.time()-t0:.1f}s")
    logger.info(
        f"\nDone in {time.time()-t_total:.1f}s total.\n"
        f"Upload {out_path} to Google Drive, then set in train_supervised.py:\n"
        f"  training_ready = '/content/drive/MyDrive/Colab Notebooks/training_ready.npz'"
    )

    # Print array summary
    print("\n=== Output arrays ===")
    data = np.load(str(out_path), allow_pickle=True)
    for k in data.files:
        if k == "metadata":
            print(f"  {k:<16}: {data[k]}")
        else:
            arr = data[k]
            print(f"  {k:<16}: shape={arr.shape}  dtype={arr.dtype}  "
                  f"min={arr.min():.4f}  max={arr.max():.4f}")


if __name__ == "__main__":
    main()