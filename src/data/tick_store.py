"""DuckDB-based tick data storage and retrieval."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl
from loguru import logger


class TickStore:
    """Persistent tick/OHLCV storage using DuckDB (embedded, columnar, fast)."""

    def __init__(self, db_path: str = "data/ticks.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                tick_volume BIGINT,
                spread INTEGER,
                PRIMARY KEY (timestamp, symbol, timeframe)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                bid DOUBLE,
                ask DOUBLE,
                last_price DOUBLE,
                volume DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_ts
            ON ohlcv (symbol, timeframe, timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticks_ts
            ON ticks (symbol, timestamp)
        """)

    def insert_ohlcv(self, df: pl.DataFrame, symbol: str, timeframe: str) -> int:
        """Insert OHLCV data, skipping duplicates."""
        if df.is_empty():
            return 0

        df = df.with_columns([
            pl.lit(symbol).alias("symbol"),
            pl.lit(timeframe).alias("timeframe"),
        ])

        # Use INSERT OR IGNORE to skip duplicates
        self.conn.execute("""
            INSERT OR IGNORE INTO ohlcv
            SELECT timestamp, symbol, timeframe, open, high, low, close, tick_volume, spread
            FROM df
        """)
        count = len(df)
        logger.info(f"Inserted up to {count} OHLCV rows for {symbol}/{timeframe}")
        return count

    def insert_ticks(self, df: pl.DataFrame, symbol: str) -> int:
        """Insert raw tick data."""
        if df.is_empty():
            return 0

        df = df.with_columns(pl.lit(symbol).alias("symbol"))
        self.conn.execute("INSERT INTO ticks SELECT * FROM df")
        logger.info(f"Inserted {len(df)} ticks for {symbol}")
        return len(df)

    def query_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """Query OHLCV data with optional time range."""
        query = """
            SELECT timestamp, open, high, low, close, tick_volume, spread
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp"

        if limit:
            query += f" LIMIT {limit}"

        result = self.conn.execute(query, params).pl()
        logger.debug(f"Queried {len(result)} OHLCV rows for {symbol}/{timeframe}")
        return result

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the most recent timestamp in the store."""
        result = self.conn.execute("""
            SELECT MAX(timestamp) as max_ts
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """, [symbol, timeframe]).fetchone()
        return result[0] if result and result[0] else None

    def get_row_count(self, symbol: str, timeframe: str) -> int:
        """Count rows for a symbol/timeframe."""
        result = self.conn.execute("""
            SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?
        """, [symbol, timeframe]).fetchone()
        return result[0] if result else 0

    def vacuum(self) -> None:
        """Optimize database storage."""
        self.conn.execute("VACUUM")

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
