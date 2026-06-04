from .data import get_mnist_loaders

__all__ = [
    "BaseInference",
    "InferenceFactory",
    "InferenceSpec",
    "MNISTInference",
    "build_arg_parser",
    "get_mnist_loaders",
    "iter_image_paths",
    "load_model",
    "load_mnist_model",
    "main",
    "predict_mnist",
    "run_inference",
    "train",
]


def __getattr__(name: str) -> object:
    if name in {
        "BaseInference",
        "InferenceFactory",
        "InferenceSpec",
        "MNISTInference",
        "build_arg_parser",
        "iter_image_paths",
        "load_model",
        "load_mnist_model",
        "main",
        "predict_mnist",
        "run_inference",
    }:
        from . import inference

        return getattr(inference, name)

    if name == "train":
        from .training import train

        return train

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
