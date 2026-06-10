from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_LOG_DIR = Path("core/logs")
_LOG_FILE = _LOG_DIR / "vigilant-detect.log"
_FMT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {extra[module]} | {message}"

_configured = False


def _setup() -> None:
    global _configured
    if _configured:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.configure(extra={"module": "vigilant-detect"})
    logger.remove()
    logger.add(sys.stdout, format=_FMT, level="DEBUG", colorize=True)
    logger.add(
        str(_LOG_FILE),
        format=_FMT,
        level="DEBUG",
        rotation="10 MB",
        retention=10,
        encoding="utf-8",
    )
    _configured = True


def get_logger(name: str = "vigilant-detect") -> logger.__class__:
    _setup()
    return logger.bind(module=name)
