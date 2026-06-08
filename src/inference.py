"""Inference base classes and MNIST inference helpers."""

from abc import ABC, abstractmethod
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

try:
    from .models import MNISTNet, SVHNNet
except ImportError:
    from models import MNISTNet, SVHNNet


class BaseInference(ABC):
    """Base class for dataset-specific inference."""

    @abstractmethod
    def predict(self, image):
        """Return a prediction for one input image."""
        raise NotImplementedError


def _default_device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_mnist_model(checkpoint_path="weights/mnist.pth", device=None):
    device = _default_device(device)
    model = MNISTNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


class MNISTInference(BaseInference):
    """MNIST inference wrapper."""

    def __init__(self, model=None, checkpoint_path="weights/mnist.pth", device=None):
        self.device = _default_device(device)
        self.model = model or load_mnist_model(checkpoint_path, self.device)
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

    def _prepare_image(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if isinstance(image, Image.Image):
            return self.transform(image).unsqueeze(0)

        raise TypeError("image must be a path or PIL image")

    def predict(self, image):
        tensor = self._prepare_image(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return int(logits.argmax(dim=1).item())


def predict_mnist(image, checkpoint_path="weights/mnist.pth", device=None):
    return MNISTInference(checkpoint_path=checkpoint_path, device=device).predict(image)


_SVHN_MEAN = (0.4377, 0.4438, 0.4728)
_SVHN_STD = (0.1980, 0.2010, 0.1970)

def load_svhn_model(checkpoint_path="weights/svhn.pth", device=None):
    device = _default_device(device)
    model = SVHNNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

class SVHNInference(BaseInference):
    def __init__(self, model=None, checkpoint_path="weights/svhn.pth", device=None):
        self.device = _default_device(device)
        self.model = model or load_svhn_model(checkpoint_path, self.device)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(_SVHN_MEAN, _SVHN_STD),
            ]
        )

    def _prepare_image(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if isinstance(image, Image.Image):
            return self.transform(image.convert("RGB")).unsqueeze(0)

        raise TypeError("image must be a path or PIL image")

    def predict(self, image):
        tensor = self._prepare_image(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return int(logits.argmax(dim=1).item())


def predict_svhn(image, checkpoint_path="weights/svhn.pth", device=None):
    return SVHNInference(checkpoint_path=checkpoint_path, device=device).predict(image)

INFERENCE_REGISTRY = {
    "mnist": MNISTInference,
    "svhn": SVHNInference,
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Classify a single image with a trained model.")
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=sorted(INFERENCE_REGISTRY.keys()),
        help="Dataset whose model to use for inference.",
    )
    parser.add_argument("--input", required=True, help="Path to the image to classify.")
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint to load (defaults to the dataset's registered path).")
    parser.add_argument("--device", default="cpu", help="Device to use, e.g. cpu, cuda, or mps.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    inference_cls = INFERENCE_REGISTRY[args.dataset]
    kwargs = {"device": args.device}
    if args.checkpoint_path:
        kwargs["checkpoint_path"] = args.checkpoint_path

    predictor = inference_cls(**kwargs)
    print(predictor.predict(args.input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BaseInference",
    "MNISTInference",
    "load_mnist_model",
    "predict_mnist",
    "SVHNInference",
    "load_svhn_model",
    "predict_svhn",
    "INFERENCE_REGISTRY",
    "build_arg_parser",
    "main",
]
