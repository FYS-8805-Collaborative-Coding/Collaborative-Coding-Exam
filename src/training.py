"""Training base classes and MNIST trainer."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn, optim

try:
    from .data import MNISTDataModule
    from .models import MNISTNet
except ImportError:
    from data import MNISTDataModule
    from models import MNISTNet


class BaseTrainer(ABC):
    """Base class for dataset-specific trainers."""

    @abstractmethod
    def train(self):
        """Train a model and return training metrics."""
        raise NotImplementedError


class MNISTTrainer(BaseTrainer):
    """Trainer for MNIST."""

    def __init__(
        self,
        data_module=None,
        model=None,
        epochs=1,
        lr=1e-3,
        checkpoint_path="weights/mnist.pth",
        device=None,
    ):
        self.data_module = data_module or MNISTDataModule()
        self.model = model or MNISTNet()
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def train(self):
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        final_loss = 0.0
        final_accuracy = 0.0

        for _ in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.data_module.train_loader():
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += batch_size

            final_loss = running_loss / total if total else 0.0
            final_accuracy = correct / total if total else 0.0

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)

        return {
            "loss": final_loss,
            "accuracy": final_accuracy,
            "epochs": self.epochs,
            "checkpoint_path": str(self.checkpoint_path),
        }


def train_mnist(
    epochs=1,
    lr=1e-3,
    batch_size=64,
    checkpoint_path="weights/mnist.pth",
    data_dir="datasets",
    num_workers=2,
    device=None,
):
    data_module = MNISTDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    trainer = MNISTTrainer(
        data_module=data_module,
        epochs=epochs,
        lr=lr,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return trainer.train()


__all__ = ["BaseTrainer", "MNISTTrainer", "train_mnist"]
