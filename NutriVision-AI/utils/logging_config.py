"""Logging configuration shared by every NutriVision AI module.

Streamlit re-executes the script on every interaction, so :func:`setup_logging`
is idempotent: it configures the root logger only once and never stacks
duplicate handlers (which would otherwise print each line many times).
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED: bool = False

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%H:%M:%S"

# Third-party libraries that are noisy at INFO level.
NOISY_LOGGERS = ("urllib3", "PIL", "asyncio", "watchdog", "matplotlib", "httpx")


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger for console output (safe to call repeatedly)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)
