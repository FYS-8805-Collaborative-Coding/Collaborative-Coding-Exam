"""Inference base classes and a dataset-agnostic predictor + factory."""

from __future__ import annotations

import argparse
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import torch
from torch import nn
from PIL import Image
from torchvision import transforms

from src.models import MNISTNet, USPSNet, SVHNNet
from src.utils import setup_logging, get_logger

logger = get_logger("inference")

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
ModelT = TypeVar("ModelT", bound=nn.Module)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaseInference(ABC):
    """Base class for dataset-specific inference."""

    @abstractmethod
    def predict(self, image: str | Path | Image.Image) -> int:
        """Return a prediction for one input image."""
        raise NotImplementedError


def _default_device(device: str | torch.device | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if path.is_absolute():
        return path

    repo_path = PROJECT_ROOT / path
    if repo_path.exists():
        return repo_path

    try:
        from importlib import resources

        bundled = resources.files(__package__ or "ccexam").joinpath(*path.parts)
        if bundled.is_file():
            return Path(str(bundled))
    except (ModuleNotFoundError, AttributeError, TypeError):
        pass

    return repo_path


def load_model(
    model_cls: type[ModelT],
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> ModelT:
    device = _default_device(device)
    model = model_cls()
    model.load_state_dict(torch.load(_resolve_checkpoint_path(checkpoint_path), map_location=device))
    model.to(device)
    model.eval()
    return model


def build_transform(
    image_size: int | tuple[int, int],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    grayscale: bool = True,
) -> Callable[[Image.Image], torch.Tensor]:
    """Build the image transform used before inference.

    When ``grayscale`` is True the image is reduced to a single channel
    (MNIST/USPS); otherwise it is converted to 3-channel RGB (SVHN).
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if grayscale:
        channel_step = transforms.Grayscale(num_output_channels=1)
    else:
        channel_step = transforms.Lambda(lambda img: img.convert("RGB"))

    return transforms.Compose(
        [
            channel_step,
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class Inference(BaseInference):
    """Generic inference runner for image classification models."""

    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[Image.Image], torch.Tensor],
        device: str | torch.device | None = None,
    ) -> None:
        self.device = _default_device(device)
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.transform = transform

    def _prepare_image(self, image: str | Path | Image.Image) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if isinstance(image, Image.Image):
            return self.transform(image).unsqueeze(0)

        raise TypeError("image must be a path or PIL image")

    def predict(self, image: str | Path | Image.Image) -> int:
        tensor = self._prepare_image(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return int(logits.argmax(dim=1).item())


@dataclass(frozen=True)
class InferenceSpec:
    model_cls: type[nn.Module]
    default_checkpoint: str
    image_size: int | tuple[int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    grayscale: bool = True
    inference_cls: type[BaseInference] = Inference


_SVHN_SPEC = InferenceSpec(
    SVHNNet,
    "weights/svhn.pth",
    (32, 32),
    (0.4377, 0.4438, 0.4728),
    (0.1980, 0.2010, 0.1970),
    grayscale=False,
)

INFERENCE_REGISTRY = {
    "mnist": InferenceSpec(MNISTNet, "weights/mnist.pth", (28, 28), (0.1307,), (0.3081,)),
    "usps": InferenceSpec(USPSNet, "weights/usps.pth", (16, 16), (0.2471,), (0.2994,)),
    "svhn": _SVHN_SPEC,
    "model-a": InferenceSpec(MNISTNet, "weights/mnist.pth", (28, 28), (0.1307,), (0.3081,)),
    "model-b": InferenceSpec(USPSNet, "weights/usps.pth", (16, 16), (0.2471,), (0.2994,)),
    "model-c": _SVHN_SPEC,
}


class InferenceFactory:
    """Create configured inference instances from a model name."""

    @classmethod
    def create(cls, model_name: str, **kwargs) -> BaseInference:
        if model_name not in INFERENCE_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        spec = INFERENCE_REGISTRY[model_name]
        device = kwargs.get("device")
        checkpoint = kwargs.get("checkpoint_path") or spec.default_checkpoint
        model = load_model(spec.model_cls, checkpoint, device)
        transform = kwargs.get("transform") or build_transform(
            image_size=kwargs.get("image_size", spec.image_size),
            mean=kwargs.get("mean", spec.mean),
            std=kwargs.get("std", spec.std),
            grayscale=kwargs.get("grayscale", spec.grayscale),
        )

        return spec.inference_cls(
            model=model,
            transform=transform,
            device=device,
        )


def iter_image_paths(input_path: str | Path) -> Iterable[Path]:
    """Yield image paths from a single image file or a directory."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        yield path
        return

    if path.is_dir():
        image_paths = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
        )
        yield from image_paths
        return

    raise FileNotFoundError(f"Input path does not exist: {path}")


def _predict(
    model: str,
    input_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[Path, int]:
    inference = InferenceFactory.create(
        model.lower(),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return {
        image_path: inference.predict(image_path)
        for image_path in iter_image_paths(input_path)
    }


def run_inference(
    model: str,
    input_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> int | list[int]:
    """Run inference and return the predicted label(s).

    Returns a single ``int`` label when ``input_path`` is one image, or a
    ``list[int]`` of labels (in sorted filename order) when it is a directory
    of images.

    Example
    -------
    >>> run_inference(model="svhn", input_path="digit.png")
    5
    >>> run_inference(model="svhn", input_path="folder_of_digits/")
    [7, 2, 1, 0]
    """
    labels = list(_predict(model, input_path, checkpoint_path, device).values())
    if Path(input_path).is_file():
        return labels[0]
    return labels


def __output_path(filename: str | Path) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        return path

    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def write_results(results: dict[Path, int], output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "prediction"])
            for image_path, prediction in results.items():
                writer.writerow([str(image_path), prediction])
    else:
        with path.open("w") as handle:
            for image_path, prediction in results.items():
                handle.write(f"{image_path}\t{prediction}\n")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for inference."""
    parser = argparse.ArgumentParser(description="Run image inference.")
    parser.add_argument(
        "--model",
        default="mnist",
        choices=sorted(INFERENCE_REGISTRY.keys()),
        help="Model or dataset alias to run.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint",
        dest="checkpoint_path",
        default=None,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example 'cpu', 'cuda', or 'mps'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Write predictions to this file (e.g. 'results/predictions.csv'). "
            "Any folders in the path are created automatically. An existing "
            "file is never overwritten, a numbered copy is created instead "
            "(predictions_1.csv, ...). Use a '.csv' or '.txt' extension."
        ),
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for inference."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    logger.info("Start  model=%s device=%s input=%s", args.model, args.device or "auto", args.input)

    results = _predict(
        model=args.model,
        input_path=args.input,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    for image_path, prediction in results.items():
        logger.info("Predicted digit: %s  (image=%s)", prediction, image_path)

    if args.output:
        output_path = write_results(results, __output_path(args.output))
        logger.info("Wrote %d prediction(s) to %s", len(results), output_path)

    logger.info("Done  %d image(s) classified", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BaseInference",
    "Inference",
    "InferenceFactory",
    "InferenceSpec",
    "build_arg_parser",
    "build_transform",
    "iter_image_paths",
    "load_model",
    "main",
    "run_inference",
    "write_results",
]
