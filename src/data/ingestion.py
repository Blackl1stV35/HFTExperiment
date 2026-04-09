"""Tick data ingestion from MT5 and CSV sources."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import polars as pl
from loguru import logger


class MT5DataSource:
    """Ingest tick and OHLCV data from MetaTrader 5."""

    TIMEFRAME_MAP = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 16385, "H4": 16388, "D1": 16408,
    }

    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "M1"):
        self.symbol = symbol
        self.timeframe = timeframe
        self._mt5 = None

    def connect(self, login: str, password: str, server: str, path: Optional[str] = None) -> bool:
        """Initialize MT5 connection."""
        try:
            import MetaTrader5 as mt5

            self._mt5 = mt5
            init_kwargs = {}
            if path:
                init_kwargs["path"] = path

            if not mt5.initialize(**init_kwargs):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False

            if not mt5.login(int(login), password=password, server=server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False

            logger.info(f"Connected to MT5: {server} (account {login})")
            return True
        except ImportError:
            logger.error("MetaTrader5 package not installed. pip install MetaTrader5")
            return False

    def fetch_ohlcv(
        self,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Fetch OHLCV bars from MT5."""
        if self._mt5 is None:
            raise RuntimeError("Not connected to MT5. Call connect() first.")

        mt5 = self._mt5
        tf = self.TIMEFRAME_MAP.get(self.timeframe)
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")

        end = end or datetime.now()
        rates = mt5.copy_rates_range(self.symbol, tf, start, end)

        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned for {self.symbol} {self.timeframe}")
            return pl.DataFrame()

        df = pl.DataFrame({
            "timestamp": [datetime.fromtimestamp(r[0]) for r in rates],
            "open": [float(r[1]) for r in rates],
            "high": [float(r[2]) for r in rates],
            "low": [float(r[3]) for r in rates],
            "close": [float(r[4]) for r in rates],
            "tick_volume": [int(r[5]) for r in rates],
            "spread": [int(r[6]) for r in rates],
        })

        logger.info(f"Fetched {len(df)} bars for {self.symbol} [{start} → {end}]")
        return df

    def fetch_ticks(self, start: datetime, end: Optional[datetime] = None) -> pl.DataFrame:
        """Fetch raw tick data from MT5."""
        if self._mt5 is None:
            raise RuntimeError("Not connected to MT5. Call connect() first.")

        mt5 = self._mt5
        end = end or datetime.now()
        ticks = mt5.copy_ticks_range(self.symbol, start, end, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) == 0:
            return pl.DataFrame()

        df = pl.DataFrame({
            "timestamp": [datetime.fromtimestamp(t[0]) for t in ticks],
            "bid": [float(t[1]) for t in ticks],
            "ask": [float(t[2]) for t in ticks],
            "last": [float(t[3]) for t in ticks],
            "volume": [float(t[4]) for t in ticks],
            "flags": [int(t[5]) for t in ticks],
        })
        return df

    def stream_ticks(self, callback: Callable[[dict], None], poll_ms: int = 100) -> None:
        """Stream live ticks with callback. Blocking call."""
        if self._mt5 is None:
            raise RuntimeError("Not connected to MT5. Call connect() first.")

        mt5 = self._mt5
        logger.info(f"Starting tick stream for {self.symbol}")
        last_time = datetime.now()

        while True:
            ticks = mt5.copy_ticks_from(self.symbol, last_time, 100, mt5.COPY_TICKS_ALL)
            if ticks is not None and len(ticks) > 0:
                for tick in ticks:
                    tick_data = {
                        "timestamp": datetime.fromtimestamp(tick[0]),
                        "bid": float(tick[1]),
                        "ask": float(tick[2]),
                        "last": float(tick[3]),
                        "volume": float(tick[4]),
                    }
                    callback(tick_data)
                last_time = datetime.fromtimestamp(ticks[-1][0]) + timedelta(milliseconds=1)

            time.sleep(poll_ms / 1000.0)

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if self._mt5:
            self._mt5.shutdown()
            logger.info("MT5 disconnected")


class CSVDataSource:
    """Load tick/OHLCV data from CSV files."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load(self, filename: str) -> pl.DataFrame:
        """Load CSV file into Polars DataFrame."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pl.read_csv(str(path), try_parse_dates=True)
        logger.info(f"Loaded {len(df)} rows from {filename}")
        return df

    def load_multiple(self, pattern: str = "*.csv") -> pl.DataFrame:
        """Load and concatenate multiple CSV files."""
        files = sorted(self.data_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {self.data_dir}")

        dfs = [pl.read_csv(str(f), try_parse_dates=True) for f in files]
        combined = pl.concat(dfs).sort("timestamp")
        logger.info(f"Loaded {len(combined)} rows from {len(files)} files")
        return combined
