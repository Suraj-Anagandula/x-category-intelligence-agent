"""Loguru-based logging setup.

Provides a single `get_logger()` entry point that configures console +
rotating file sinks once (idempotently) and returns the shared logger.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure loguru sinks. Safe to call multiple times; only runs once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        log_dir / "scraper_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="00:00",
        retention="14 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    _CONFIGURED = True


def get_logger():
    """Return the shared, configured loguru logger."""
    if not _CONFIGURED:
        from app.config import settings

        configure_logging(settings.log_dir, settings.log_level)
    return logger
