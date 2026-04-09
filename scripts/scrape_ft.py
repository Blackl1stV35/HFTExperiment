"""Scrape FT.com articles via sitemaps for training sentiment data.

Usage:
    # Scrape recent 6 months (requires FT cookie string in .env or --cookies)
    python scripts/scrape_ft.py --start 2024-10 --end 2025-03

    # With explicit cookie string
    python scripts/scrape_ft.py --start 2025-01 --end 2025-03 \\
        --cookies "FTSession=abc123; spoor-id=xyz789"

    # Scrape with lower relevance threshold (more articles, less focused)
    python scripts/scrape_ft.py --start 2024-01 --min-relevance 0.005

    # Build FinBERT embeddings from cached articles
    python scripts/scrape_ft.py --build-embeddings --data-dir data
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data.ft_scraper import FTSitemapScraper, FTArticleCache
from src.utils.logger import setup_logger

from dotenv import load_dotenv
load_dotenv()

async def scrape(args):
    """Run FT sitemap scraping pipeline (async — uses Playwright for articles)."""
    # Resolve cookie string: CLI arg > env var > empty (sitemap-only mode)
    cookie_string = args.cookies or os.getenv("FT_COOKIE_STRING", "")
    if not cookie_string:
        logger.warning(
            "No FT cookie string provided. Article body extraction will be limited.\n"
            "Set FT_COOKIE_STRING in .env or pass --cookies '...'\n"
            "To get cookies: open FT.com in Chrome → DevTools → Application → Cookies\n"
            "Copy the full cookie header string."
        )

    scraper = FTSitemapScraper(
        cookie_string=cookie_string,
        cache_dir=args.cache_dir,
        request_delay=(args.min_delay, args.max_delay),
    )

    try:
        articles = await scraper.fetch_articles(
            start_month=args.start,
            end_month=args.end,
            min_relevance=args.min_relevance,
            max_articles_per_month=args.max_per_month,
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping complete: {len(articles)} relevant articles")
        logger.info(f"Date range: {args.start} → {args.end}")
        logger.info(f"Cache dir: {args.cache_dir}")

        if articles:
            # Show top 10 by relevance
            top = sorted(articles, key=lambda a: a.relevance_score, reverse=True)[:10]
            logger.info(f"\nTop 10 most relevant articles:")
            for i, a in enumerate(top, 1):
                logger.info(f"  {i}. [{a.relevance_score:.3f}] {a.headline[:80]}")

            # Stats
            avg_relevance = sum(a.relevance_score for a in articles) / len(articles)
            with_body = sum(1 for a in articles if a.body_text)
            logger.info(f"\nStats:")
            logger.info(f"  Average relevance: {avg_relevance:.4f}")
            logger.info(f"  Articles with body text: {with_body}/{len(articles)}")
            logger.info(f"  Date range: {articles[0].published_at[:10]} → {articles[-1].published_at[:10]}")

        # Also save as Parquet for easy Polars/Pandas access
        if articles:
            try:
                import polars as pl

                output_dir = Path(args.cache_dir) / "parquet"
                output_dir.mkdir(parents=True, exist_ok=True)
                df = pl.DataFrame([a.to_dict() for a in articles])
                parquet_path = output_dir / f"ft_{args.start}_to_{args.end or 'now'}.parquet"
                df.write_parquet(parquet_path)
                logger.info(f"Saved Parquet: {parquet_path}")
            except ImportError:
                pass

    finally:
        await scraper.close()


def build_embeddings(args):
    """Build FinBERT embeddings from cached FT articles."""
    import numpy as np
    import polars as pl

    from src.data.sentiment import TrainingSentimentBuilder
    from src.data.tick_store import TickStore

    # Load price data timestamps
    store = TickStore(f"{args.data_dir}/ticks.duckdb")
    df = store.query_ohlcv("XAUUSD", "M1")
    store.close()

    if df.is_empty():
        logger.error("No price data. Run download_data.py first.")
        return

    timestamps = df["timestamp"].to_list()
    logger.info(f"Building embeddings for {len(timestamps)} bars")

    # Build
    builder = TrainingSentimentBuilder(
        ft_cache_dir=f"{args.cache_dir}/processed",
        model_device="cuda" if args.gpu else "cpu",
        lookback_hours=24,
    )

    embeddings = builder.build_embedding_series(timestamps, batch_interval=15)

    # Save
    output_path = Path(args.data_dir) / "sentiment_embeddings.npy"
    np.save(str(output_path), embeddings)
    logger.info(f"Saved embeddings to {output_path}: shape={embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="FT.com article scraper for training data")
    subparsers = parser.add_subparsers(dest="command")

    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape FT articles")
    scrape_parser.add_argument("--start", required=True, help="Start month YYYY-MM")
    scrape_parser.add_argument("--end", default=None, help="End month YYYY-MM")
    scrape_parser.add_argument("--min-relevance", type=float, default=0.01)
    scrape_parser.add_argument("--max-per-month", type=int, default=500)
    scrape_parser.add_argument("--cache-dir", default="data/ft_cache")
    scrape_parser.add_argument("--min-delay", type=float, default=1.0)
    scrape_parser.add_argument("--max-delay", type=float, default=3.0)
    scrape_parser.add_argument("--cookies", default=None, help="FT.com cookie string")

    # Build embeddings command
    embed_parser = subparsers.add_parser("embed", help="Build FinBERT embeddings")
    embed_parser.add_argument("--cache-dir", default="data/ft_cache")
    embed_parser.add_argument("--data-dir", default="data")
    embed_parser.add_argument("--gpu", action="store_true")

    # Shortcut: no subcommand = scrape
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-relevance", type=float, default=0.01)
    parser.add_argument("--max-per-month", type=int, default=500)
    parser.add_argument("--cache-dir", default="data/ft_cache")
    parser.add_argument("--min-delay", type=float, default=1.0)
    parser.add_argument("--max-delay", type=float, default=3.0)
    parser.add_argument("--cookies", default=None, help="FT.com cookie string")
    parser.add_argument("--build-embeddings", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()

    setup_logger()

    if args.command == "embed" or getattr(args, "build_embeddings", False):
        build_embeddings(args)
    elif args.command == "scrape" or args.start:
        if not args.start:
            parser.error("--start is required for scraping")
        asyncio.run(scrape(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
