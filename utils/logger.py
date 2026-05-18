import sys
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = ["logger"]


def setup_logger() -> Any:
    # Remove default logger
    logger.remove()

    # Define custom format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{file}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Add console logger
    logger.add(sys.stderr, format=log_format, level="INFO")

    # Add file logger
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        str(log_dir / "app_{time}.log"),
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention="1 week",
    )

    return logger


# Initialize default logger
setup_logger()
