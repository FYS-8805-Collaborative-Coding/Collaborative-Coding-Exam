"""Top-level package entry points.

``run_inference`` and ``train`` are exposed lazily via :func:`__getattr__` so
importing the package does not pull torch and the model machinery unless one
of those callables is actually used.
"""


def __getattr__(name: str) -> object:
    """Lazily import and return :func:`run_inference` or :func:`train`."""
    if name == "run_inference":
        from .inference import run_inference

        return run_inference

    if name == "train":
        from .training import train

        return train

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
