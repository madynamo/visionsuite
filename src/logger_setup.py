"""
logger_setup.py
----------------
Centralised logging configuration for VisionSuite.

Every module in the project calls `get_logger(__name__)` instead of creating
its own handlers. This keeps log formatting consistent and makes it possible
to redirect all logs to a single rotating file for monitoring purposes
(Non-Functional Requirement: Logging & Monitoring).
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "logs")
LOG_FILE = os.path.join(LOG_DIR, "visionsuite.log")

_CONFIGURED = False


def _configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger exactly once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Rotate at 2 MB, keep 3 backups, so logs never grow unbounded
    # (Non-Functional Requirement: Resource Efficiency).
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with consistent formatting/handlers."""
    _configure_root_logger(level)
    return logging.getLogger(name)
