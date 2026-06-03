"""Training base classes and a dataset-agnostic trainer + factory.

This module provides a generic `Trainer` that implements the shared training
loop, a small `MNISTTrainer` wrapper for backward compatibility, a
`TrainerFactory`/registry for mapping dataset names to data/model classes,
and a convenience `train(...)` entry point that selects the right classes
based on a dataset name.
"""

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional, Callable

import torch
from torch import nn, optim

try:
    from .data import MNISTDataModule, USPSDataModule
    from .models import MNISTNet, USPSNet
except ImportError:
    from data import MNISTDataModule, USPSDataModule
    from models import MNISTNet, USPSNet


class BaseTrainer(ABC):
    """Base class for dataset-specific trainers."""

    @abstractmethod
    def train(self):
        """Train a model and return training metrics."""
        raise NotImplementedError


class Trainer(BaseTrainer):
    """Generic trainer implementing the common training loop.

    Parameters are intentionally simple: the trainer expects an already
    constructed `data_module` and `model` so it can be reused by different
    dataset/model combinations.
    """

    def __init__(
        self,
        data_module,
        model,
        epochs: int = 1,
        lr: float = 1e-3,
        checkpoint_path: str = "weights/model.pth",
        device: str | None = None,
        loss_fn: Optional[Callable] = None,
    ):
        self.data_module = data_module
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Allow customizing the loss function; default to CrossEntropyLoss
        self.loss_fn = loss_fn or getattr(torch.nn, "CrossEntropyLoss")()

    def train(self):
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = self.loss_fn
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


class MNISTTrainer(Trainer):
    """Backward-compatible MNIST trainer wrapper."""

    def __init__(
        self,
        data_module=None,
        model=None,
        epochs: int = 1,
        lr: float = 1e-3,
        checkpoint_path: str = "weights/mnist.pth",
        device: str | None = None,
        loss_fn: Optional[Callable] = None,
    ):
        data_module = data_module or MNISTDataModule()
        model = model or MNISTNet()
        super().__init__(data_module, model, epochs=epochs, lr=lr, checkpoint_path=checkpoint_path, device=device, loss_fn=loss_fn)


@dataclass
class DatasetSpec:
    data_module_cls: Type
    model_cls: Type
    default_checkpoint: str
    trainer_cls: Type = Trainer


# Registry mapping dataset name -> DatasetSpec. Add SVHN/USPS entries as they
# become available in `src.data` and `src.models`.
DATASET_REGISTRY = {
    "mnist": DatasetSpec(MNISTDataModule, MNISTNet, "weights/mnist.pth", trainer_cls=MNISTTrainer),
    "usps":  DatasetSpec(USPSDataModule, USPSNet, "weights/usps.pth"),
}

# Mapping of CLI loss name -> loss class attribute name (resolved at runtime)
LOSS_MAP = {
    "cross_entropy": "CrossEntropyLoss",
    "mse": "MSELoss",
}


class TrainerFactory:
    """Create configured trainer instances from a dataset name."""

    @classmethod
    def create(cls, dataset_name: str, **kwargs):
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        spec = DATASET_REGISTRY[dataset_name]

        data_module = spec.data_module_cls(
            data_dir=kwargs.get("data_dir", "datasets"),
            batch_size=kwargs.get("batch_size", 64),
            num_workers=kwargs.get("num_workers", 2),
        )

        model = spec.model_cls()

        trainer_cls = spec.trainer_cls or Trainer
        checkpoint = kwargs.get("checkpoint_path") or spec.default_checkpoint

        return trainer_cls(
            data_module=data_module,
            model=model,
            epochs=kwargs.get("epochs", 1),
            lr=kwargs.get("lr", 1e-3),
            checkpoint_path=checkpoint,
            device=kwargs.get("device"),
            loss_fn=kwargs.get("loss_fn"),
        )


def train(dataset: str = "mnist", **kwargs):
    """Convenience entry point: pick dataset and run training.

    Example: `train(dataset="mnist", epochs=2, batch_size=64)`
    """
    trainer = TrainerFactory.create(dataset, **kwargs)
    return trainer.train()


def build_arg_parser():
    """Build the command-line argument parser for training."""
    parser = argparse.ArgumentParser(description="Train a dataset model.")
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Dataset to train on.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--checkpoint-path", default=None, help="Where to save the model checkpoint.")
    parser.add_argument("--data-dir", default="datasets", help="Directory containing the datasets.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--device", default=None, help="Device to use, e.g. cpu or cuda.")
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "mse"],
        default=None,
        help="Loss function to use (defaults to CrossEntropyLoss when unspecified).",
    )
    return parser


def main(argv=None):
    """Command-line entry point for training."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Map CLI loss name to a loss instance (or None to use trainer default)
    loss_cls_name = LOSS_MAP.get(args.loss)
    loss_fn = None
    if loss_cls_name is not None:
        loss_cls = getattr(nn, loss_cls_name)
        loss_fn = loss_cls()

    result = train(
        dataset=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        device=args.device,
        loss_fn=loss_fn,
    )

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BaseTrainer",
    "Trainer",
    "MNISTTrainer",
    "DatasetSpec",
    "TrainerFactory",
    "build_arg_parser",
    "main",
    "train",
]
