"""Download historical data from MT5 or generate synthetic data for testing.

Usage:
    python scripts/download_data.py --symbol XAUUSD --timeframe M1 --days 365
    python scripts/download_data.py --synthetic --days 30
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
from loguru import logger

from src.data.tick_store import TickStore
from src.utils.config import load_env, BrokerConfig
from src.utils.logger import setup_logger


def generate_synthetic_data(
    days: int = 30,
    timeframe_minutes: int = 1,
    base_price: float = 2000.0,
    volatility: float = 0.0005,
) -> pl.DataFrame:
    """Generate synthetic XAUUSD M1 data for testing.

    Uses geometric Brownian motion with mean-reversion to simulate
    realistic-looking gold price movements.
    """
    bars_per_day = int(24 * 60 / timeframe_minutes)
    n_bars = days * bars_per_day

    logger.info(f"Generating {n_bars} synthetic bars ({days} days of M{timeframe_minutes})")

    np.random.seed(42)

    # Geometric Brownian motion with mean reversion
    prices = np.zeros(n_bars)
    prices[0] = base_price
    mean_price = base_price

    for i in range(1, n_bars):
        # Mean reversion component
        reversion = 0.001 * (mean_price - prices[i - 1])
        # Random walk
        shock = np.random.normal(0, volatility * prices[i - 1])
        # Regime shifts (occasional large moves)
        if np.random.random() < 0.001:
            shock *= 5
        prices[i] = prices[i - 1] + reversion + shock

    # Generate OHLCV from close prices
    timestamps = [
        datetime(2024, 1, 1) + timedelta(minutes=i * timeframe_minutes)
        for i in range(n_bars)
    ]

    # Filter out weekends
    valid = [i for i, ts in enumerate(timestamps) if ts.weekday() < 5]
    timestamps = [timestamps[i] for i in valid]
    prices = prices[valid]

    n = len(prices)
    noise = np.abs(np.random.normal(0, 0.3, n))

    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": prices + np.random.normal(0, 0.1, n),
        "high": prices + noise,
        "low": prices - noise,
        "close": prices,
        "tick_volume": np.random.randint(50, 500, n).astype(int),
        "spread": np.random.randint(15, 35, n).astype(int),
    })

    logger.info(
        f"Synthetic data: {len(df)} bars, "
        f"price range [{df['close'].min():.2f}, {df['close'].max():.2f}]"
    )
    return df


def download_from_mt5(
    symbol: str,
    timeframe: str,
    days: int,
    broker_config: BrokerConfig,
) -> pl.DataFrame:
    """Download historical data from MT5."""
    from src.data.ingestion import MT5DataSource

    source = MT5DataSource(symbol, timeframe)
    connected = source.connect(
        login=broker_config.login,
        password=broker_config.password,
        server=broker_config.server,
        path=broker_config.path,
    )

    if not connected:
        logger.error("Failed to connect to MT5")
        return pl.DataFrame()

    end = datetime.now()
    start = end - timedelta(days=days)

    df = source.fetch_ohlcv(start, end)
    source.disconnect()
    return df


def main():
    parser = argparse.ArgumentParser(description="Download XAUUSD historical data")
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
    parser.add_argument("--timeframe", default="M1", help="Timeframe (M1, M5, H1, etc)")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()

    setup_logger()

    if args.synthetic:
        df = generate_synthetic_data(days=args.days)
    else:
        load_env()
        broker_config = BrokerConfig.from_env()
        if not broker_config.login:
            logger.error("MT5 credentials not set. Use --synthetic for test data.")
            logger.info("Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env file")
            sys.exit(1)
        df = download_from_mt5(args.symbol, args.timeframe, args.days, broker_config)

    if df.is_empty():
        logger.error("No data retrieved")
        sys.exit(1)

    # Save to DuckDB
    store = TickStore(f"{args.output}/ticks.duckdb")
    store.insert_ohlcv(df, args.symbol, args.timeframe)
    count = store.get_row_count(args.symbol, args.timeframe)
    store.close()

    # Also save as CSV for convenience
    csv_path = Path(args.output) / f"{args.symbol}_{args.timeframe}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(csv_path))

    logger.info(f"Data saved: {count} rows in DuckDB + CSV at {csv_path}")


if __name__ == "__main__":
    main()
