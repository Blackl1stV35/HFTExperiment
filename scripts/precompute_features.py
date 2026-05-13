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
    features        (N, 12)  float32  — scaled OHLCV + 4 micro + session_phase + rq_regime
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
    tick_volume_raw (N,)     int32    — raw bar tick count (for VIO / inventory_pressure)
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
    parser.add_argument("--csv",         default=None,
                        help="Direct path to OHLCV CSV (bypasses TickStore/DuckDB)")
    parser.add_argument("--existing-npz", default=None,
                        help="Previous training_ready.npz to copy rq from if not in regime CSV")
    parser.add_argument("--symbol",      default="XAUUSD")
    parser.add_argument("--timeframe",   default="M1")
    parser.add_argument("--regime-csv",  default="data/regime/daily_regime_labels.csv")
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--scaler",      default="window_minmax")
    parser.add_argument("--atr-period",  type=int, default=14)
    parser.add_argument("--atr-mult-tp", type=float, default=1.5)
    parser.add_argument("--atr-mult-sl", type=float, default=0.75)  # kept asymmetric — Bear regime bias dominates
    parser.add_argument("--atr-min-tp",  type=float, default=150.0)
    parser.add_argument("--atr-max-tp",  type=float, default=800.0)
    parser.add_argument("--max-holding", type=int, default=40)
    parser.add_argument("--pip-value",   type=float, default=0.10)
    parser.add_argument("--output",      default="data/training_ready.npz")
    args = parser.parse_args()

    t_total = time.time()

    # ── 1. Load raw M1 data ──────────────────────────────────────────────────
    t0 = time.time()
    if args.csv:
        import polars as pl
        logger.info(f"Loading from CSV: {args.csv}")
        df = pl.read_csv(args.csv, try_parse_dates=True, infer_schema_length=10000)
        df = df.rename({c: c.lower() for c in df.columns})
        # Normalise common column name variants
        rename_map = {}
        for src_c, dst_c in [
            ("time","timestamp"),("date","timestamp"),("datetime","timestamp"),
            ("vol","tick_volume"),("volume","tick_volume"),
            ("tickvol","tick_volume"),("tick_vol","tick_volume"),
        ]:
            if src_c in df.columns and dst_c not in df.columns:
                rename_map[src_c] = dst_c
        if rename_map:
            df = df.rename(rename_map)
        if "spread" not in df.columns:
            df = df.with_columns(pl.lit(6).cast(pl.Int32).alias("spread"))
        if "tick_volume" not in df.columns:
            df = df.with_columns(pl.lit(50).cast(pl.Int64).alias("tick_volume"))
        if df["timestamp"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("timestamp").str.to_datetime(strict=False))
        logger.info(f"  {len(df):,} bars loaded from CSV in {time.time()-t0:.1f}s")
    else:
        logger.info(f"Loading {args.symbol} {args.timeframe} from {args.data_dir}/ticks.duckdb")
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
    assert len(df_wd) == len(features) + ws, \
        f"Weekday df length {len(df_wd)} != features {len(features)+ws}"

    # ── 5b. tick_volume_raw for inventory_pressure VIO (Phase 7/8) ────────────
    # Raw unscaled bar tick count. VIO = Σ(V×sign) / Σ(V) requires absolute
    # magnitudes — MinMax or tanh scaling destroys the normalisation property.
    logger.info("Extracting tick_volume_raw...")
    if "tick_volume" in df_wd.columns:
        tv_full = df_wd["tick_volume"].to_numpy().astype(np.int32)
        tick_volume_raw = tv_full[ws:]
        assert len(tick_volume_raw) == n, \
            f"tick_volume_raw {len(tick_volume_raw)} != n {n}"
        logger.info(f"  tick_volume_raw: min={tick_volume_raw.min()} "
                    f"max={tick_volume_raw.max()} mean={tick_volume_raw.mean():.1f}")
    else:
        tick_volume_raw = np.zeros(n, dtype=np.int32)
        logger.warning("No tick_volume column — tick_volume_raw=zeros. "
                       "inventory_pressure VIO will not be computable.")

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

    # If rq is all-constant (0.5 placeholder from missing regime_quality_norm),
    # attempt to copy from --existing-npz which has correctly computed rq values.
    if rq.std() < 0.01 and args.existing_npz:
        try:
            _old = np.load(args.existing_npz, allow_pickle=True)
            if "rq" in _old and len(_old["rq"]) == n:
                rq = _old["rq"].astype(np.float64)
                logger.info(f"rq: loaded from existing NPZ (std={rq.std():.4f})")
            else:
                logger.warning(f"rq: existing NPZ has wrong length or no rq column")
        except Exception as e:
            logger.warning(f"rq: could not load from existing NPZ: {e}")
    elif rq.std() < 0.01:
        logger.warning(
            "rq: all values are 0.5 (regime_quality_norm missing from regime CSV). "
            "Pass --existing-npz data/training_ready.npz to copy correct rq values. "
            "Feature[11] will be uninformative without this."
        )
    else:
        logger.info(f"rq: from regime CSV (std={rq.std():.4f} mean={rq.mean():.4f})")

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
        "feature_dim":  features.shape[1],  # 11: 6-OHLCV + 4-micro + session_phase
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
        session_phase   = session_phase.astype(np.float32),
        tick_volume_raw = tick_volume_raw,          # int32 raw bar tick count
        metadata        = np.array(metadata),
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


# ── Phase 6 bypass features ───────────────────────────────────────────────────
# Run phase6_feature_exploration.ipynb first to validate KS/MI/redundancy.
# Pass validated feature names via --bypass-features flag.

def compute_bypass_features(close, high, low, timestamps_ns, bypass_list, dxy_parquet=None):
    """Compute and RobustScale validated Phase 6 bypass features.

    Returns ndarray (N, len(bypass_list)) float32.
    """
    import numpy as np
    from sklearn.preprocessing import RobustScaler

    def _wilder_atr(h, l, c, p=14):
        n  = len(c)
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        tr = np.concatenate([[tr[0]], tr])
        atr = np.empty(n); atr[0] = tr[0]; a = 1.0/p
        for i in range(1, n): atr[i] = atr[i-1]*(1-a)+tr[i]*a
        return atr

    def _rolling_zscore(arr, w=120, eps=1e-8):
        arr = arr.astype(np.float64); n = len(arr); out = np.empty(n, np.float32)
        for i in range(n):
            win = arr[max(0, i-w+1):i+1]; out[i] = float(np.tanh((arr[i]-win.mean())/(win.std()+eps)))
        return out

    def _mtf_return(c, window):
        lr = np.concatenate([[0.0], np.log(c[1:]/(c[:-1]+1e-8))])
        return _rolling_zscore(np.convolve(lr, np.ones(window), mode='same'))

    out = []
    for name in bypass_list:
        if name == 'vwap_dev_norm':
            tp   = (high+low+close)/3.0
            vwap = np.cumsum(tp)/np.arange(1, len(tp)+1)
            atr  = _wilder_atr(high, low, close)
            feat = np.tanh((close-vwap)/(atr+1e-8)).astype(np.float32)
        elif name in ('ret_5m', 'ret_15m', 'ret_1h'):
            feat = _mtf_return(close, {'ret_5m':5,'ret_15m':15,'ret_1h':60}[name])
        elif name == 'dxy_return_20':
            import polars as pl
            dxy  = pl.read_parquet(dxy_parquet)
            dc   = dxy['close'].to_numpy().astype(np.float64)
            dt   = dxy['timestamp'].cast(pl.Datetime).to_numpy().astype(np.int64)
            idx  = np.clip(np.searchsorted(dt, timestamps_ns.astype(np.int64), side='right')-1, 0, len(dc)-1)
            lr   = np.concatenate([[0.0], np.log(dc[idx][1:]/(dc[idx][:-1]+1e-8))])
            feat = _rolling_zscore(np.convolve(lr, np.ones(20)/20.0, mode='same'))
        else:
            raise ValueError(f"Unknown bypass feature: {name}")
        out.append(feat.reshape(-1))

    if not out:
        return np.zeros((len(close), 0), dtype=np.float32)
    arr = np.stack(out, axis=1).astype(np.float32)
    return RobustScaler().fit_transform(arr).astype(np.float32)