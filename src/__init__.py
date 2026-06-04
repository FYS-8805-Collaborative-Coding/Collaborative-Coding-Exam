from .data import get_mnist_loaders

__all__ = [
    "get_mnist_loaders",
    "run_inference",
    "train",
]


def __getattr__(name: str) -> object:
    if name == "run_inference":
        from .inference import run_inference

        return run_inference

    if name == "train":
        from .training import train

        return train

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
