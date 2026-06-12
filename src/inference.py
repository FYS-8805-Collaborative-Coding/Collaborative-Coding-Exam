"""Inference base classes and a dataset-agnostic predictor + factory."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import transforms

from src.constants import DATASET_STATS

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

    @abstractmethod
    def predict_batch(self, images: Iterable[str | Path | Image.Image], batch_size: int = 32) -> list[int]:
        """Return a list of predictions for multiple input images."""
        raise NotImplementedError
    
    @abstractmethod
    def predict_batch_tensor(self, batch: torch.Tensor) -> list[int]:
        """Optional method for running inference on an already-transformed batch tensor (B, C, H, W)."""
        raise NotImplementedError("predict_batch_tensor is not implemented for this inference class")

def _default_device(device: str | torch.device | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_model(
    model_cls: type[ModelT],
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    **model_kwargs,
) -> ModelT:
    device = _default_device(device)
    model = model_cls(**model_kwargs)
    model.to(device)
    model.load_state_dict(torch.load(_resolve_checkpoint_path(checkpoint_path), map_location=device))
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

    def _ensure_size(self, x: torch.Tensor) -> torch.Tensor:
        """Internal safety check to resize input if it doesn't match the model's expected size."""
        expected = getattr(self.model, "input_size", None)
        if expected and x.shape[-2:] != (expected, expected):
            return F.interpolate(
                x, size=(expected, expected), mode="bilinear", align_corners=False
            )
        return x

    def _process_batch(self, batch_tensors: list[torch.Tensor]) -> list[int]:
        """Concatenate tensors and run a single forward pass."""
        input_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
        input_tensor = self._ensure_size(input_tensor)
        with torch.no_grad():
            logits = self.model(input_tensor)
        return logits.argmax(dim=1).tolist()

    def predict(self, image: str | Path | Image.Image) -> int:
        tensor = self._prepare_image(image).to(self.device)
        tensor = self._ensure_size(tensor)
        with torch.no_grad():
            logits = self.model(tensor)
        return int(logits.argmax(dim=1).item())
    
    def predict_batch_tensor(self, batch: torch.Tensor) -> list[int]:
        """Run inference on an already-transformed batch tensor (B, C, H, W)."""
        batch = self._ensure_size(batch.to(self.device))
        with torch.no_grad():
            logits = self.model(batch)
        return logits.argmax(dim=1).tolist()

    def predict_batch(self, images: Iterable[str | Path | Image.Image], batch_size: int = 32) -> list[int]:
        all_predictions = []
        current_batch = []

        for img in images:
            current_batch.append(self._prepare_image(img))
            if len(current_batch) >= batch_size:
                all_predictions.extend(self._process_batch(current_batch))
                current_batch = []

        if current_batch:
            all_predictions.extend(self._process_batch(current_batch))

        return all_predictions


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
    image_size=DATASET_STATS["svhn"]["image_size"],
    **DATASET_STATS["svhn"],
    grayscale=DATASET_STATS["svhn"]["grayscale"],
)

INFERENCE_REGISTRY = {
    "mnist": InferenceSpec(
        MNISTNet, "weights/mnist.pth",
        image_size=DATASET_STATS["mnist"]["image_size"],
        mean=DATASET_STATS["mnist"]["mean"],
        std=DATASET_STATS["mnist"]["std"],
        grayscale=DATASET_STATS["mnist"]["grayscale"]
    ),
    "usps": InferenceSpec(
        USPSNet, "weights/usps.pth",
        image_size=DATASET_STATS["usps"]["image_size"],
        mean=DATASET_STATS["usps"]["mean"],
        std=DATASET_STATS["usps"]["std"],
        grayscale=DATASET_STATS["usps"]["grayscale"]
    ),
    "svhn": _SVHN_SPEC,
    "model-a": InferenceSpec(
        MNISTNet, "weights/mnist.pth",
        image_size=DATASET_STATS["mnist"]["image_size"],
        mean=DATASET_STATS["mnist"]["mean"],
        std=DATASET_STATS["mnist"]["std"],
        grayscale=DATASET_STATS["mnist"]["grayscale"]
    ),
    "model-b": InferenceSpec(
        USPSNet, "weights/usps.pth",
        image_size=DATASET_STATS["usps"]["image_size"],
        mean=DATASET_STATS["usps"]["mean"],
        std=DATASET_STATS["usps"]["std"],
        grayscale=DATASET_STATS["usps"]["grayscale"]
    ),
    "model-c": _SVHN_SPEC,
}


class InferenceFactory:
    """Create configured inference instances from a model name."""

    # ... (no changes needed in create method, as it already uses spec attributes)
    # The `create` method already correctly uses `spec.image_size`, `spec.mean`,
    # `spec.std`, and `spec.grayscale` to build the transform and load the model.

    @classmethod
    def create(cls, model_name: str, **kwargs) -> BaseInference:
        if model_name not in INFERENCE_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        spec = INFERENCE_REGISTRY[model_name]
        device = kwargs.get("device")
        checkpoint = kwargs.get("checkpoint_path") or spec.default_checkpoint

        # Extract input size to pass to model constructor, handling both int and tuple
        img_size = kwargs.get("image_size", spec.image_size)
        input_size = img_size[0] if isinstance(img_size, tuple) else img_size

        model = load_model(spec.model_cls, checkpoint, device, input_size=input_size)
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
        if not image_paths:
            raise ValueError(f"No supported image files found in {path}")
        yield from image_paths
        return

    raise FileNotFoundError(f"Input path does not exist: {path}")


def run_inference(
    model: str,
    input_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[Path, int]:
    """Run inference for the requested model over one image or image directory."""
    inference = InferenceFactory.create(
        model.lower(),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    image_paths = list(iter_image_paths(input_path))
    predictions = inference.predict_batch(image_paths, batch_size=batch_size)

    return {
        path: pred for path, pred in zip(image_paths, predictions)
    }


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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example 'cpu', 'cuda', or 'mps'.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for inference."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    logger.info("Start  model=%s device=%s input=%s", args.model, args.device or "auto", args.input)

    results = run_inference(
        model=args.model,
        input_path=args.input,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    for image_path, prediction in results.items():
        logger.info("Predicted digit: %s  (image=%s)", prediction, image_path)
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
]
