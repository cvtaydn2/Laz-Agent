from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("editor_agent")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
