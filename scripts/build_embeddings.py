"""Build daily consensus FinBERT sentiment embeddings from cached FT.com articles.

Reads the FT article cache and the OHLCV tick database, computes one 768-dim
embedding per calendar day, then broadcasts each day's embedding to all M1 bars
on that day. Saves the result as data/sentiment_embeddings.npy.

Usage:
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --cache-dir data/ft_cache --gpu
    python scripts/build_embeddings.py --lookback 48  # 48h article window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from src.data.sentiment import TrainingSentimentBuilder
from src.data.tick_store import TickStore
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Build daily consensus sentiment embeddings"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--cache-dir", default="data/ft_cache/processed",
                        help="FT article cache directory")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--lookback", type=int, default=24,
                        help="Article lookback window in hours")
    parser.add_argument("--max-articles", type=int, default=30,
                        help="Max articles per day for consensus")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for FinBERT inference")
    parser.add_argument("--output", default=None,
                        help="Output path (default: data/sentiment_embeddings.npy)")
    args = parser.parse_args()

    setup_logger()

    # Load price data to get timestamps
    db_path = f"{args.data_dir}/ticks.duckdb"
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}. Run download_data.py first.")
        sys.exit(1)

    store = TickStore(db_path)
    df = store.query_ohlcv(args.symbol, args.timeframe)
    store.close()

    if df.is_empty():
        logger.error("No price data found in database.")
        sys.exit(1)

    timestamps = df["timestamp"].to_list()
    logger.info(f"Price data: {len(timestamps)} M1 bars")
    logger.info(f"Date range: {timestamps[0]} → {timestamps[-1]}")

    # Build daily embeddings
    builder = TrainingSentimentBuilder(
        ft_cache_dir=args.cache_dir,
        model_device="cuda" if args.gpu else "cpu",
        lookback_hours=args.lookback,
        max_articles_per_day=args.max_articles,
    )

    daily_cache_path = f"{args.data_dir}/daily_embedding_cache.npy"
    embeddings = builder.build_embedding_series(
        timestamps,
        cache_path=daily_cache_path,
    )

    # Save
    output_path = args.output or f"{args.data_dir}/sentiment_embeddings.npy"
    np.save(output_path, embeddings)
    logger.info(f"Saved: {output_path} — shape {embeddings.shape}")

    # Verification
    nonzero_pct = (np.abs(embeddings).sum(axis=1) > 0).mean() * 100
    unique_embeddings = len(set(map(tuple, embeddings)))
    logger.info(
        f"Verification: {nonzero_pct:.1f}% of bars have non-zero embeddings, "
        f"{unique_embeddings} unique daily vectors"
    )


if __name__ == "__main__":
    main()
