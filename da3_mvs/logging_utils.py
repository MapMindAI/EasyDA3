from __future__ import annotations

import logging

_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


class ColorLineFormatter(logging.Formatter):
    def __init__(self, use_color: bool = True):
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d | %(message)s",
            datefmt="%H:%M:%S",
        )
        self.use_color = bool(use_color)

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)

        color = _LEVEL_COLORS.get(record.levelno, "")
        if not color:
            return super().format(record)

        original_levelname = record.levelname
        record.levelname = f"{color}{record.levelname}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def configure_logging(level: int | str = logging.INFO, use_color: bool = True) -> None:
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO

    handler = logging.StreamHandler()
    handler.setFormatter(ColorLineFormatter(use_color=use_color))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def ensure_logging_configured(level: int | str = logging.INFO, use_color: bool = True) -> None:
    if logging.getLogger().handlers:
        return

    configure_logging(level=level, use_color=use_color)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
