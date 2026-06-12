import logging

# Aligned columns keep multi-line output tidy:
#   2026-06-08 10:49 | INFO    | training   | epoch 1/20  loss=0.82  acc=0.73
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-10s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level="INFO"):
    """Configure root logging once with a consistent, readable format.

    Parameters
    ----------
    level : str or int
        Level name (e.g. ``"INFO"``, ``"DEBUG"``) or a ``logging`` constant.
        Unknown names fall back to ``INFO``.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    return logging.getLogger()


def get_logger(name):
    """Return a named logger. Call :func:`setup_logging` in ``main`` first."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
