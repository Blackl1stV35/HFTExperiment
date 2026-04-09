"""Centralized logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_dir: str = "logs",
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> None:
    """Configure loguru logger for the project."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler for all logs
    logger.add(
        str(log_path / "trading_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="gz",
    )

    # Separate file for errors
    logger.add(
        str(log_path / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="gz",
    )

    # Separate file for trades
    logger.add(
        str(log_path / "trades_{time:YYYY-MM-DD}.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        filter=lambda record: "trade" in record["extra"],
        rotation="1 day",
        retention="90 days",
    )


trade_logger = logger.bind(trade=True)
