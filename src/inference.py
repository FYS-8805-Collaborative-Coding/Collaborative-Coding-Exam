"""Inference base classes and a dataset-agnostic predictor + factory."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*", category=UserWarning)

import argparse
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import transforms

from .constants import DATASET_STATS

from .models import MNISTNet, USPSNet, SVHNNet
from .utils import (
    ascii_digit_to_image,
    get_logger,
    is_ascii_image,
    read_ascii_image,
    setup_logging,
)

logger = get_logger("inference")

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
ModelT = TypeVar("ModelT", bound=nn.Module)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Inputs prefixed with this scheme resolve to a packaged sample image, e.g.
# `samples:svhn_digit_5.png` -> ccexam/samples/svhn_digit_5.png
SAMPLE_PREFIX = "samples:"
_SAMPLES_SUBDIR = ("samples",)


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
    **model_kwargs,
) -> ModelT:
    device = _default_device(device)
    model = model_cls(**model_kwargs)
    model.load_state_dict(torch.load(_resolve_checkpoint_path(checkpoint_path), map_location=device, weights_only=True))
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
            path = Path(image)
            image = ascii_digit_to_image(path) if is_ascii_image(path) else Image.open(path)

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
    image_size=DATASET_STATS["svhn"]["image_size"], # Explicitly pass
    mean=DATASET_STATS["svhn"]["mean"],             # Explicitly pass
    std=DATASET_STATS["svhn"]["std"],               # Explicitly pass
    grayscale=DATASET_STATS["svhn"]["grayscale"],   # Explicitly pass
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


def _is_image(path: Path) -> bool:
    """Return True if ``path`` is a decodable image, judged by its content.

    The file extension is ignored: an image with no extension (e.g. ``digit``)
    is accepted. Non-image is rejected because PIL cannot decode it.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _is_supported_input(path: Path) -> bool:
    """Return True for decodable images or valid ASCII digit images."""
    return _is_image(path) or is_ascii_image(path)


def _samples_dir():
    """Return a path-like to the packaged sample-images directory."""
    return resources.files(__package__ or "ccexam").joinpath(*_SAMPLES_SUBDIR)


def list_samples() -> list[str]:
    """Return the available packaged sample filenames (images + ASCII), sorted."""
    allowed = IMAGE_EXTENSIONS | {".txt"}
    try:
        return sorted(
            entry.name
            for entry in _samples_dir().iterdir()
            if Path(entry.name).suffix.lower() in allowed
        )
    except Exception:
        return []


def _resolve_sample(name: str) -> Path:
    """Resolve a sample name (e.g. ``svhn_digit_5`` or ``svhn_digit_5.png``) to
    a packaged image path, trying the name as given and then with ``.png``."""
    samples = _samples_dir()
    candidates = [name] if Path(name).suffix else [f"{name}.png", name]
    for candidate in candidates:
        try:
            entry = samples.joinpath(candidate)
            if entry.is_file():
                return Path(str(entry))
        except (FileNotFoundError, NotADirectoryError):
            continue
    available = ", ".join(list_samples()) or "none"
    raise FileNotFoundError(f"Unknown sample: '{name}'. Available samples: {available}")


def _resolve_input(input_path: str | Path) -> str | Path:
    """Map a ``samples:NAME`` input to a packaged image; pass everything else
    through unchanged so normal file/directory paths still work."""
    if isinstance(input_path, str) and input_path.startswith(SAMPLE_PREFIX):
        return _resolve_sample(input_path[len(SAMPLE_PREFIX):])
    return input_path


def iter_image_paths(input_path: str | Path) -> Iterable[Path]:
    """Yield image or ASCII-image paths from a single file or a directory."""
    path = Path(input_path)
    if path.is_file():
        if not _is_supported_input(path):
            raise ValueError(f"Not a valid image or ASCII image file: {path}")
        yield path
        return

    if path.is_dir():
        image_paths = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and _is_supported_input(candidate)
        )
        if not image_paths:
            raise ValueError(f"No valid image or ASCII image files found in {path}")
        yield from image_paths
        return

    raise FileNotFoundError(f"Input path does not exist: {path}")


def _predict(
    model: str,
    input_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[Path, int] | int:
    input_path = _resolve_input(input_path)
    inference = InferenceFactory.create(
        model.lower(),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    path = Path(input_path)
    if path.is_file() and is_ascii_image(path):
        return inference.predict(path)
    image_paths = list(iter_image_paths(input_path))
    predictions = inference.predict_batch(image_paths, batch_size=batch_size)

    return {
        path: pred for path, pred in zip(image_paths, predictions)
    }


def run_inference(
    model: str,
    input_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[Path, int]:
    """Run inference on a single image or a directory of images.

    Returns a dictionary mapping image paths to predicted integer labels.
    """
    return _predict(
        model=model,
        input_path=input_path,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
    )


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
        help=(
            "Path to a single image file OR a directory of images, OR a packaged "
            "sample as 'samples:NAME' (e.g. 'samples:svhn_digit_5.png'). Images "
            "are detected by content, and ASCII digit files are accepted as .txt "
            "or .ascii files using '#', 'X', '1', '@', or '*' for strokes and "
            "'.', space, '0', or '-' for background. "
            "Missing paths and unsupported files are reported as invalid input."
        ),
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

    try:
        results = run_inference(
            model=args.model,
            input_path=args.input,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            batch_size=args.batch_size, # Pass batch_size
        )
    except FileNotFoundError as exc:
        logger.error("Invalid input: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        return 1

    if not results:
        logger.error("Invalid input: no valid images could be processed from '%s'", args.input)
        return 1

    for path, prediction in results.items():
        logger.info("Predicted digit: %s  (image=%s)", prediction, path)

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
    "ascii_digit_to_image",
    "build_arg_parser",
    "build_transform",
    "iter_image_paths",
    "is_ascii_image",
    "load_model",
    "main",
    "read_ascii_image",
    "run_inference",
    "write_results",
]
