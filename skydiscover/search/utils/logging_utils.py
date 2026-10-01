"""Logging utilities for the search subsystem.

Provides a compact console formatter, a noise-reducing filter, and a
one-call helper to wire up both file and console logging for a
discovery run.
"""

import logging
import os
import time

_QUIET = {"route", "server"}

# Libraries that install their own handler and so bypass _ConsoleFilter. Their
# records would otherwise follow the root level, which setup_search_logging pins
# at DEBUG so the file handler captures everything.
_SELF_HANDLED = ("jax", "jaxlib")


class _ConsoleFormatter(logging.Formatter):
    """Compact single-line formatter: ``HH:MM:SS [module] message``."""

    def format(self, record):
        ts = self.formatTime(record, "%H:%M:%S")
        name = (
            record.name[len("skydiscover.") :]
            if record.name.startswith("skydiscover.")
            else record.name
        )
        parts = name.split(".")
        short = f"search.{parts[1]}" if parts[0] == "search" and len(parts) >= 3 else parts[-1]
        fmt = (
            f"{ts} {record.levelname} [{short}] "
            if record.levelno >= logging.WARNING
            else f"{ts} [{short}] "
        )
        return fmt + record.getMessage()


class _ConsoleFilter(logging.Filter):
    """Only pass skydiscover messages, suppressing noisy modules below WARNING."""

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        if not record.name.startswith("skydiscover") or record.name.split(".")[-1] in _QUIET:
            return False
        return True


def setup_search_logging(log_level: str, log_dir: str, name: str) -> None:
    """Configure root logger with a timestamped file handler and a console handler.

    The file handler always captures DEBUG+, while the console handler
    respects *log_level* so the terminal stays clean by default.
    """
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # JAX attaches a stderr handler to its own loggers on import, so its records
    # never reach the console handler's filter. Pinning the level here — before
    # an evaluated program imports jax — both silences that handler and becomes
    # the default jax derives for its own logging config.
    for logger_name in _SELF_HANDLED:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    console_level = getattr(logging, log_level)

    log_file = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(_ConsoleFormatter())
        ch.addFilter(_ConsoleFilter())
        root.addHandler(ch)

    logging.getLogger(__name__).debug(f"Logging to {log_file}")
