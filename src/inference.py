"""Inference base classes and registry helpers."""

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar

import torch
from torch import nn
from PIL import Image
from torchvision import transforms

try:
    from .models import MNISTNet
except ImportError:
    from models import MNISTNet

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
ModelT = TypeVar("ModelT", bound=nn.Module)


class BaseInference(ABC):
    """Base class for dataset-specific inference."""

    @abstractmethod
    def predict(self, image: str | Path | Image.Image) -> int:
        """Return a prediction for one input image."""
        raise NotImplementedError


def _default_device(device: str | torch.device | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_model(
    model_cls: type[ModelT],
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> ModelT:
    device = _default_device(device)
    model = model_cls()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


class MNISTInference(BaseInference):
    """MNIST inference wrapper."""

    def __init__(
        self,
        model: nn.Module | None = None,
        checkpoint_path: str | Path = "weights/mnist.pth",
        device: str | torch.device | None = None,
    ) -> None:
        self.device = _default_device(device)
        self.model = model or load_model(MNISTNet, checkpoint_path, self.device)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

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


@dataclass
class InferenceSpec:
    model_cls: type[nn.Module]
    default_checkpoint: str
    inference_cls: type[BaseInference] = MNISTInference


INFERENCE_REGISTRY = {
    "mnist": InferenceSpec(MNISTNet, "weights/mnist.pth", inference_cls=MNISTInference),
    "model-a": InferenceSpec(MNISTNet, "weights/mnist.pth", inference_cls=MNISTInference),
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

        return spec.inference_cls(
            model=model,
            checkpoint_path=checkpoint,
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
) -> dict[Path, int]:
    """Run inference for the requested model over one image or image directory."""
    inference = InferenceFactory.create(
        model.lower(),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return {
        image_path: inference.predict(image_path)
        for image_path in iter_image_paths(input_path)
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for inference."""
    parser = argparse.ArgumentParser(description="Run MNIST image inference.")
    parser.add_argument(
        "--model",
        default="model-a",
        choices=sorted(INFERENCE_REGISTRY.keys()),
        help="Model alias to run. Use 'model-a' or 'mnist'.",
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
        help="Path to the MNIST model checkpoint.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example 'cpu', 'cuda', or 'mps'.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for inference."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    results = run_inference(
        model=args.model,
        input_path=args.input,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    for image_path, prediction in results.items():
        print(f"{image_path}: {prediction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_inference",
]
