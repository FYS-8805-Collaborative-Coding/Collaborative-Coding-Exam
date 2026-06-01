"""Inference base classes and MNIST inference helpers."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

try:
    from .models import MNISTNet
except ImportError:
    from models import MNISTNet


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


__all__ = ["BaseInference", "MNISTInference", "load_mnist_model", "predict_mnist"]
