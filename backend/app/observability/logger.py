"""Lightweight application logging. Never log settings or credentials."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    base = logging.getLogger("salesops")
    if not base.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        base.addHandler(handler)
        base.setLevel(logging.INFO)
        base.propagate = False
    return logging.getLogger(f"salesops.{name}")
