"""Structured logging helper."""
from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                                               "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(os.getenv("SPAMGUARD_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger
