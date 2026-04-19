r"""
Ingest HistData.com M1 ZIP archives into the DuckDB TickStore.
It reads directly from the ZIP files, parses the MetaTrader ASCII format,
and seamlessly injects the history into the pipeline.

Usage:
    python scripts/ingest_histdata.py --zip-dir "C:\Users\user\Downloads"
"""

import argparse
import zipfile
import sys
from pathlib import Path

import polars as pl
from loguru import logger

# Ensure we can import the project's data layer
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.tick_store import TickStore


def process_histdata_zip(zip_path: Path) -> pl.DataFrame:
    """Reads a HistData ZIP file in-memory and formats it for the pipeline."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the CSV file inside the zip (ignores readmes/text files)
        csv_filename = next((f for f in z.namelist() if f.endswith('.csv')), None)
        if not csv_filename:
            logger.warning(f"No CSV found in {zip_path.name}. Skipping.")
            return pl.DataFrame()

        # Read CSV bytes directly from the zip archive into Polars
        with z.open(csv_filename) as f:
            try:
                # HistData MT format has no headers: Date, Time, O, H, L, C, V
                df = pl.read_csv(
                    f.read(),
                    has_header=False,
                    new_columns=["date", "time", "open", "high", "low", "close", "tick_volume"],
                    separator=",", # Default HistData separator
                    ignore_errors=True
                )
            except Exception as e:
                logger.error(f"Failed to parse {zip_path.name}: {e}")
                return pl.DataFrame()

    if df.is_empty():
        return df

    # HistData time can sometimes be "0:00" instead of "00:00". We pad it to ensure clean parsing.
    df = df.with_columns(
        pl.col("time").str.pad_start(5, '0')
    )

    # Combine 'date' (YYYY.MM.DD) and 'time' (HH:MM) into a Polars Datetime
    df = df.with_columns(
        (pl.col("date") + " " + pl.col("time"))
        .str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M", strict=False)
        .alias("timestamp")
    )

    # Mock the 'spread' column (required by Phase 3 RL architecture)
    df = df.with_columns(pl.lit(20).alias("spread"))

    # Enforce standard pipeline column order and drop any parsing failures
    required_cols = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]
    df = df.select(required_cols).drop_nulls(subset=["timestamp"])
    
    # Ensure chronologically sorted and unique
    df = df.sort("timestamp").unique(subset=["timestamp"])

    return df


def main():
    p = argparse.ArgumentParser(description="Ingest HistData ZIPs into DuckDB")
    p.add_argument("--zip-dir", required=True, help="Directory containing the downloaded .zip files")
    p.add_argument("--symbol", default="XAUUSD", help="Asset symbol")
    p.add_argument("--timeframe", default="M1", help="Timeframe (e.g., M1)")
    p.add_argument("--output", default="data", help="Output directory for DuckDB")
    args = p.parse_args()

    dir_path = Path(args.zip_dir)
    
    # Grab all XAUUSD M1 ZIPs in the directory
    zip_files = sorted(dir_path.glob("HISTDATA_COM_MT_XAUUSD_M1*.zip"))

    if not zip_files:
        logger.error(f"No HistData ZIP files found in {args.zip_dir}")
        logger.info("Ensure the files are named 'HISTDATA_COM_MT_XAUUSD_M1_*.zip'")
        return

    logger.info(f"Found {len(zip_files)} ZIP archives. Initializing database...")
    
    # Connect to DuckDB
    store = TickStore(f"{args.output}/ticks.duckdb")

    total_inserted = 0
    for zf in zip_files:
        logger.info(f"Cracking open {zf.name}...")
        df = process_histdata_zip(zf)
        
        if not df.is_empty():
            store.insert_ohlcv(df, args.symbol, args.timeframe)
            total_inserted += len(df)
            logger.info(f" -> Inserted {len(df):,} rows from {zf.name} [{df['timestamp'].min().date()} to {df['timestamp'].max().date()}]")

    # Final DB Report
    final_count = store.get_row_count(args.symbol, args.timeframe)
    store.close()
    
    logger.info(f"{'═'*55}")
    logger.info(f"SUCCESS: DuckDB now contains {final_count:,} total M1 bars for {args.symbol}.")
    logger.info(f"{'═'*55}")


if __name__ == "__main__":
    main()