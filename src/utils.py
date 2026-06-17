import logging
from pathlib import Path

from PIL import Image

# Aligned columns keep multi-line output tidy:
#   2026-06-08 10:49 | INFO    | training   | epoch 1/20  loss=0.82  acc=0.73
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-10s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
ASCII_BACKGROUND = "."
ASCII_FOREGROUND = "#"
ASCII_BACKGROUND_CHARS = frozenset({ASCII_BACKGROUND, " ", "0", "-"})
ASCII_FOREGROUND_CHARS = frozenset({ASCII_FOREGROUND, "X", "1", "@", "*"})
ASCII_IMAGE_CHARS = ASCII_BACKGROUND_CHARS | ASCII_FOREGROUND_CHARS
ASCII_IMAGE_EXTENSIONS = frozenset({".ascii", ".txt"})


def read_ascii_image(path: str | Path) -> list[str]:
    """Read and validate a fixed-width ASCII digit image."""
    image_path = Path(path)
    if image_path.suffix.lower() not in ASCII_IMAGE_EXTENSIONS:
        raise ValueError("ASCII image must use a .txt or .ascii extension")

    try:
        lines = image_path.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("ASCII image must contain ASCII characters only") from exc

    if not lines:
        raise ValueError("ASCII image is empty")

    width = len(lines[0])
    if width == 0:
        raise ValueError("ASCII image rows cannot be empty")

    if any(len(line) != width for line in lines):
        raise ValueError("ASCII image rows must all have the same width")

    invalid_chars = sorted(
        {char for line in lines for char in line if char not in ASCII_IMAGE_CHARS}
    )
    if invalid_chars:
        raise ValueError(f"ASCII image contains invalid characters: {invalid_chars}")

    if not any(char in ASCII_FOREGROUND_CHARS for line in lines for char in line):
        raise ValueError("ASCII image must contain at least one foreground pixel")

    return lines


def is_ascii_image(path: str | Path) -> bool:
    """Return True when ``path`` is a valid ASCII digit image."""
    try:
        read_ascii_image(path)
    except (OSError, ValueError):
        return False
    return True


def ascii_digit_to_image(path: str | Path) -> Image.Image:
    """Convert a valid ASCII digit image to a grayscale PIL image."""
    lines = read_ascii_image(path)
    width = len(lines[0])
    height = len(lines)
    pixels = [
        255 if char in ASCII_FOREGROUND_CHARS else 0
        for line in lines
        for char in line
    ]

    image = Image.new("L", (width, height))
    image.putdata(pixels)
    return image


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


__all__ = [
    "ascii_digit_to_image",
    "get_logger",
    "is_ascii_image",
    "read_ascii_image",
    "setup_logging",
]
