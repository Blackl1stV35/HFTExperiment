"""Download historical OHLCV data from MT5 (chunked, resume-capable) or yfinance.

MT5 SAFETY — 6666-day M1 requests:
    copy_rates_range() is safe but brokers cap M1 history at 2-5 years.
    Requesting 18 years of M1 in one call silently truncates.
    Solution: chunk into CHUNK_DAYS=30 windows + 0.5s sleep between requests.
    Resume support reads last stored timestamp from DuckDB, skipping chunks
    already downloaded. Empty chunks (beyond broker history) are logged and
    skipped gracefully — they never abort the run.
    For pre-2015 M1 data use --source yfinance (Yahoo Finance, free, unlimited).

yfinance interval limits:
    1m  → 7d max     5m/15m/30m → 60d max
    1h  → 730d max   1d         → unlimited (back to 2000+)
    For full 6666-day deep history use --timeframe D1 with yfinance.

Ticker mapping (MT5 → yfinance):
    XAUUSD  GC=F      XAGUSD SI=F    XTIUSD CL=F   XCUUSD HG=F
    EURUSD  EURUSD=X  US500 ^GSPC    DXY DX-Y.NYB
    USD10Y  ^TNX      USDT10Y ^TIP

Usage:
    python scripts/download_data.py --symbol XAUUSD --days 6666
    python scripts/download_data.py --source yfinance --regime-tickers --days 6666 --timeframe D1
    python scripts/download_data.py --synthetic --days 30
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
from loguru import logger

from src.data.tick_store import TickStore
from src.utils.config import load_env, BrokerConfig
from src.utils.logger import setup_logger


# ── Ticker maps ────────────────────────────────────────────────────────────────

MT5_TO_YF: dict[str, str] = {
    "XAUUSD":  "GC=F",
    "XAGUSD":  "SI=F",
    "XTIUSD":  "CL=F",
    "XCUUSD":  "HG=F",
    "EURUSD":  "EURUSD=X",
    "US500":   "^GSPC",
    "DXY":     "DX-Y.NYB",
    "USD10Y":  "^TNX",
    "USDT10Y": "^TIP",
}
REGIME_TICKERS = list(MT5_TO_YF.keys())

MT5_TF_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408,
}
BARS_PER_DAY = {
    "M1": 1440, "M5": 288, "M15": 96, "M30": 48,
    "H1": 24,   "H4": 6,   "D1": 1,
}

CHUNK_DAYS = 30          # window size per MT5 request
MT5_CHUNK_SLEEP = 0.5    # seconds between MT5 requests


# ── Synthetic ──────────────────────────────────────────────────────────────────

def generate_synthetic_data(days=30, tf_minutes=1, base=2000.0, vol=0.0005):
    bpd = int(24 * 60 / tf_minutes)
    n = days * bpd
    logger.info(f"Generating {n} synthetic bars ({days}d M{tf_minutes})")
    np.random.seed(42)
    p = np.zeros(n); p[0] = base
    for i in range(1, n):
        p[i] = p[i-1] + 0.001*(base-p[i-1]) + np.random.normal(0, vol*p[i-1]) * (1 + 4*(np.random.random()<0.001))
    ts = [datetime(2024,1,1)+timedelta(minutes=i*tf_minutes) for i in range(n)]
    valid = [i for i,t in enumerate(ts) if t.weekday()<5]
    ts = [ts[i] for i in valid]; p = p[valid]; n = len(p)
    noise = np.abs(np.random.normal(0, 0.3, n))
    df = pl.DataFrame({
        "timestamp": ts,
        "open": p+np.random.normal(0,.1,n), "high": p+noise,
        "low": p-noise, "close": p,
        "tick_volume": np.random.randint(50,500,n).astype(int),
        "spread": np.random.randint(15,35,n).astype(int),
    })
    logger.info(f"Synthetic: {len(df)} bars, ${df['close'].min():.0f}–${df['close'].max():.0f}")
    return df


# ── MT5 chunked download ───────────────────────────────────────────────────────

def download_from_mt5_chunked(symbol, timeframe, days, broker_config, resume_from=None):
    """30-day chunked MT5 download with resume + empty-chunk tolerance."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5 not installed. Use --source yfinance.")
        return pl.DataFrame()

    kw = {"path": broker_config.path} if broker_config.path else {}
    if not mt5.initialize(**kw):
        logger.error(f"MT5 init failed: {mt5.last_error()}"); return pl.DataFrame()
    if not mt5.login(int(broker_config.login), password=broker_config.password, server=broker_config.server):
        logger.error(f"MT5 login failed: {mt5.last_error()}"); mt5.shutdown(); return pl.DataFrame()
    logger.info(f"MT5 connected: {broker_config.server}")

    tf = MT5_TF_MAP.get(timeframe)
    if tf is None:
        logger.error(f"Unsupported timeframe: {timeframe}"); mt5.shutdown(); return pl.DataFrame()

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    exp_bars = BARS_PER_DAY.get(timeframe, 1) * CHUNK_DAYS * 5 // 7

    # Build chunk list
    chunks, cursor = [], start_dt
    while cursor < end_dt:
        chunks.append((cursor, min(cursor + timedelta(days=CHUNK_DAYS), end_dt)))
        cursor = chunks[-1][1]

    logger.info(f"MT5: {symbol}/{timeframe}, {days}d → {len(chunks)} chunks of {CHUNK_DAYS}d")
    if days > 1000:
        logger.warning(
            f"{days} days of {timeframe} requested. Most brokers keep only 2-5yr of M1. "
            "Chunks beyond broker history return 0 bars (skipped silently). "
            "Use --source yfinance for pre-2019 deep history."
        )

    all_dfs, skipped, empty = [], 0, 0
    for i, (cs, ce) in enumerate(chunks, 1):
        if resume_from and ce <= resume_from:
            skipped += 1; continue
        rates = mt5.copy_rates_range(symbol, tf, cs, ce)
        if rates is None or len(rates) == 0:
            empty += 1
            if empty <= 3 or i % 50 == 0:
                logger.warning(f"  Chunk {i}/{len(chunks)} [{cs.date()}→{ce.date()}]: 0 bars")
            time.sleep(MT5_CHUNK_SLEEP); continue
        cov = len(rates) / max(exp_bars, 1)
        if (i % 20 == 0 or i == len(chunks)):
            logger.info(f"  Chunk {i}/{len(chunks)} [{cs.date()}→{ce.date()}]: {len(rates):,} bars ({cov:.0%} expected)")
        elif cov < 0.5 and timeframe == "M1":
            logger.warning(f"  Chunk {i}/{len(chunks)} [{cs.date()}→{ce.date()}]: only {cov:.0%} coverage")
        chunk_df = pl.DataFrame({
            "timestamp":   [datetime.fromtimestamp(r[0]) for r in rates],
            "open":        [float(r[1]) for r in rates],
            "high":        [float(r[2]) for r in rates],
            "low":         [float(r[3]) for r in rates],
            "close":       [float(r[4]) for r in rates],
            "tick_volume": [int(r[5]) for r in rates],
            "spread":      [int(r[6]) for r in rates],
        })
        all_dfs.append(chunk_df)
        time.sleep(MT5_CHUNK_SLEEP)

    mt5.shutdown()
    logger.info(f"MT5 done: {len(all_dfs)} chunks with data, {skipped} resumed, {empty} empty")
    if not all_dfs:
        logger.warning("No data from MT5. Try --source yfinance for deeper history.")
        return pl.DataFrame()
    df = pl.concat(all_dfs).sort("timestamp").unique(subset=["timestamp"])
    logger.info(f"MT5 combined: {len(df):,} bars for {symbol}/{timeframe}")
    return df


# ── yfinance download ──────────────────────────────────────────────────────────

def download_from_yfinance(symbol, days, timeframe="D1", resume_from=None):
    """yfinance download — all 9 regime tickers, back to 2000+ at D1."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. pip install yfinance"); return pl.DataFrame()

    yf_symbol = MT5_TO_YF.get(symbol, symbol)
    TF_INTERVAL = {"M1":"1m","M5":"5m","M15":"15m","M30":"30m","H1":"1h","H4":"4h","D1":"1d"}
    TF_MAX = {"1m":7,"5m":60,"15m":60,"30m":60,"1h":730,"4h":730,"1d":99999}

    interval = TF_INTERVAL.get(timeframe, "1d")
    max_days = TF_MAX.get(interval, 99999)
    if days > max_days:
        logger.warning(f"yfinance '{interval}' max lookback is {max_days}d (requested {days}d). Capping.")
        days = max_days

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    if resume_from and resume_from > start_dt:
        start_dt = resume_from + timedelta(days=1)
        logger.info(f"Resume from {start_dt.date()}")

    try:
        hist = yf.Ticker(yf_symbol).history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=interval, auto_adjust=True,
        )
    except Exception as e:
        logger.error(f"yfinance failed for {yf_symbol}: {e}"); return pl.DataFrame()

    if hist.empty:
        logger.warning(f"yfinance: no data for {yf_symbol}"); return pl.DataFrame()

    hist = hist.reset_index()
    ts_col = "Datetime" if "Datetime" in hist.columns else "Date"
    # Remove timezone if present
    try:
        hist[ts_col] = hist[ts_col].dt.tz_localize(None)
    except Exception:
        try:
            hist[ts_col] = hist[ts_col].dt.tz_convert(None)
        except Exception:
            pass

    df = pl.DataFrame({
        "timestamp":   list(hist[ts_col].values),
        "open":        hist["Open"].astype(float).tolist(),
        "high":        hist["High"].astype(float).tolist(),
        "low":         hist["Low"].astype(float).tolist(),
        "close":       hist["Close"].astype(float).tolist(),
        "tick_volume": hist["Volume"].fillna(0).astype(int).tolist(),
        "spread":      [20] * len(hist),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime))

    # Drop weekends
    df = df.filter(pl.col("timestamp").dt.weekday().is_in([0,1,2,3,4]))
    logger.info(f"yfinance {symbol}({yf_symbol}): {len(df):,} bars [{df['timestamp'].min()} → {df['timestamp'].max()}]")
    return df


# ── Save ───────────────────────────────────────────────────────────────────────

def save_and_report(df, symbol, timeframe, output_dir):
    if df.is_empty():
        logger.warning(f"Nothing to save for {symbol}/{timeframe}"); return
    store = TickStore(f"{output_dir}/ticks.duckdb")
    store.insert_ohlcv(df, symbol, timeframe)
    count = store.get_row_count(symbol, timeframe)
    store.close()
    csv_path = Path(output_dir) / f"{symbol}_{timeframe}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(csv_path))
    logger.info(f"Saved {symbol}/{timeframe}: {count:,} total rows | CSV → {csv_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Download OHLCV — MT5 chunked or yfinance")
    p.add_argument("--symbol",         default="XAUUSD")
    p.add_argument("--timeframe",      default="M1")
    p.add_argument("--days",           type=int, default=6666)
    p.add_argument("--source",         default="mt5", choices=["mt5","yfinance"])
    p.add_argument("--synthetic",      action="store_true")
    p.add_argument("--regime-tickers", action="store_true",
                   help="Download all 9 regime tickers (DXY, USD10Y, XAGUSD, ...)")
    p.add_argument("--output",         default="data")
    p.add_argument("--no-resume",      action="store_true",
                   help="Re-download even if data already exists in DB")
    args = p.parse_args()
    setup_logger()

    if args.synthetic:
        df = generate_synthetic_data(days=args.days)
        save_and_report(df, "XAUUSD", "M1", args.output)
        return

    symbols = REGIME_TICKERS if args.regime_tickers else [args.symbol]
    if args.regime_tickers:
        logger.info(f"Regime tickers: {symbols}")
        if args.source == "mt5":
            logger.warning("DXY/USD10Y/US500 may not be available on all MT5 brokers. "
                           "Consider --source yfinance for these.")

    broker_config = None
    if args.source == "mt5":
        load_env()
        broker_config = BrokerConfig.from_env()
        if not broker_config.login:
            logger.error("MT5 credentials missing. Add MT5_LOGIN/PASSWORD/SERVER to .env")
            sys.exit(1)

    results = {}
    for symbol in symbols:
        logger.info(f"{'─'*55}")
        logger.info(f"Symbol: {symbol} | Source: {args.source} | TF: {args.timeframe} | Days: {args.days}")
        store = TickStore(f"{args.output}/ticks.duckdb")
        resume_from = None if args.no_resume else store.get_latest_timestamp(symbol, args.timeframe)
        store.close()
        if resume_from:
            logger.info(f"  Resume from {resume_from}")

        if args.source == "mt5":
            df = download_from_mt5_chunked(symbol, args.timeframe, args.days, broker_config, resume_from)
        else:
            df = download_from_yfinance(symbol, args.days, args.timeframe, resume_from)

        save_and_report(df, symbol, args.timeframe, args.output)
        results[symbol] = len(df)

    logger.info(f"{'═'*55}")
    logger.info("DOWNLOAD COMPLETE")
    for sym, n in results.items():
        logger.info(f"  {sym:<12} → {MT5_TO_YF.get(sym,'?'):<12}: {n:>8,} new bars")
    logger.info(f"{'═'*55}")


if __name__ == "__main__":
    main()
