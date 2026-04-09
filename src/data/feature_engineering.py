"""Feature engineering: technical indicators and derived features."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger


def add_ta_indicators(df: pl.DataFrame, indicators: list[str]) -> pl.DataFrame:
    """Add technical analysis indicators to OHLCV DataFrame.

    Uses pure Polars expressions where possible, falls back to pandas_ta
    for complex indicators. All indicators are computed in a single pass.
    """
    # Convert to pandas for pandas_ta (polars doesn't have TA library)
    pdf = df.to_pandas()

    try:
        import pandas_ta as ta
    except ImportError:
        logger.error("pandas_ta not installed. pip install pandas_ta")
        return df

    for ind in indicators:
        try:
            if ind == "rsi_14":
                pdf["rsi_14"] = ta.rsi(pdf["close"], length=14)
            elif ind == "bb_20_2":
                bb = ta.bbands(pdf["close"], length=20, std=2)
                if bb is not None:
                    pdf["bb_upper"] = bb.iloc[:, 0]
                    pdf["bb_mid"] = bb.iloc[:, 1]
                    pdf["bb_lower"] = bb.iloc[:, 2]
                    pdf["bb_width"] = (pdf["bb_upper"] - pdf["bb_lower"]) / pdf["bb_mid"]
                    pdf["bb_pctb"] = (pdf["close"] - pdf["bb_lower"]) / (
                        pdf["bb_upper"] - pdf["bb_lower"] + 1e-8
                    )
            elif ind == "atr_14":
                pdf["atr_14"] = ta.atr(pdf["high"], pdf["low"], pdf["close"], length=14)
            elif ind == "ema_9":
                pdf["ema_9"] = ta.ema(pdf["close"], length=9)
            elif ind == "ema_21":
                pdf["ema_21"] = ta.ema(pdf["close"], length=21)
            elif ind == "macd_12_26_9":
                macd = ta.macd(pdf["close"], fast=12, slow=26, signal=9)
                if macd is not None:
                    pdf["macd"] = macd.iloc[:, 0]
                    pdf["macd_signal"] = macd.iloc[:, 1]
                    pdf["macd_hist"] = macd.iloc[:, 2]
            elif ind == "adx_14":
                adx = ta.adx(pdf["high"], pdf["low"], pdf["close"], length=14)
                if adx is not None:
                    pdf["adx_14"] = adx.iloc[:, 0]
                    pdf["dmp_14"] = adx.iloc[:, 1]
                    pdf["dmn_14"] = adx.iloc[:, 2]
            elif ind == "stoch_14_3":
                stoch = ta.stoch(pdf["high"], pdf["low"], pdf["close"], k=14, d=3)
                if stoch is not None:
                    pdf["stoch_k"] = stoch.iloc[:, 0]
                    pdf["stoch_d"] = stoch.iloc[:, 1]
            elif ind == "vwap":
                # Approximate VWAP using tick_volume
                if "tick_volume" in pdf.columns:
                    typical = (pdf["high"] + pdf["low"] + pdf["close"]) / 3
                    pdf["vwap"] = (typical * pdf["tick_volume"]).cumsum() / pdf[
                        "tick_volume"
                    ].cumsum()
            elif ind == "obv":
                pdf["obv"] = ta.obv(pdf["close"], pdf["tick_volume"])
            else:
                logger.warning(f"Unknown indicator: {ind}")
        except Exception as e:
            logger.warning(f"Failed to compute {ind}: {e}")

    # Convert back to Polars
    result = pl.from_pandas(pdf)
    logger.info(f"Added {len(indicators)} TA indicators. Columns: {result.columns}")
    return result


def compute_microstructure_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute tick microstructure features useful for short-term prediction."""
    return df.with_columns([
        # Price returns
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("return_1"),
        (pl.col("close") / pl.col("close").shift(5) - 1).alias("return_5"),
        (pl.col("close") / pl.col("close").shift(20) - 1).alias("return_20"),

        # Candle body and wick ratios
        ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-8)).alias(
            "body_ratio"
        ),
        ((pl.col("high") - pl.col("close").clip(pl.col("open"), None))
         / (pl.col("high") - pl.col("low") + 1e-8)).alias("upper_wick_ratio"),

        # Realized volatility (rolling)
        (pl.col("close") / pl.col("close").shift(1) - 1)
        .rolling_std(window_size=20)
        .alias("volatility_20"),

        # Volume features
        (pl.col("tick_volume") / pl.col("tick_volume").rolling_mean(window_size=20)).alias(
            "volume_ratio"
        ),

        # Spread features (normalized)
        (pl.col("spread").cast(pl.Float64) / pl.col("close")).alias("spread_pct"),

        # High-low range (normalized)
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct"),
    ])


def compute_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute features specifically for regime classification."""
    return df.with_columns([
        # Trend strength: slope of EMA over N bars
        (pl.col("close").rolling_mean(window_size=50)
         - pl.col("close").rolling_mean(window_size=50).shift(20)).alias("trend_strength"),

        # Volatility regime
        (pl.col("close") / pl.col("close").shift(1) - 1)
        .rolling_std(window_size=60)
        .alias("volatility_60"),

        # Mean reversion signal: distance from rolling mean
        ((pl.col("close") - pl.col("close").rolling_mean(window_size=50))
         / pl.col("close").rolling_std(window_size=50)).alias("mean_reversion_z"),

        # Volume profile
        (pl.col("tick_volume").rolling_mean(window_size=20)
         / pl.col("tick_volume").rolling_mean(window_size=100)).alias("volume_profile"),
    ])


def select_features(df: pl.DataFrame, feature_list: list[str]) -> np.ndarray:
    """Extract specified features as numpy array, dropping NaN rows."""
    available = [c for c in feature_list if c in df.columns]
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        logger.warning(f"Missing features (skipped): {missing}")

    result = df.select(available).drop_nulls().to_numpy().astype(np.float32)
    logger.info(f"Selected {len(available)} features, {result.shape[0]} samples")
    return result
