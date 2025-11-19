"""Central logging configuration for the Agentic AI Workshop pipeline."""
from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Iterable

LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
DEFAULT_LOG_FILE = LOGS_DIR / "workshop.log"


def configure_logging(*, level: int = logging.INFO, extra_handlers: Iterable[logging.Handler] | None = None) -> Logger:
    """Configure application-wide logging with both console and file handlers."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid duplicate handlers when re-configuring
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(DEFAULT_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    for handler in extra_handlers or []:
        logger.addHandler(handler)

    logger.debug("Logging configured. Writing to %s", DEFAULT_LOG_FILE)
    return logger
