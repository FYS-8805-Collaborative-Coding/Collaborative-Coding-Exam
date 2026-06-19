"""Training base classes and a dataset-agnostic trainer + factory.

This module provides a generic `Trainer` that implements the shared training
loop, a small `MNISTTrainer` wrapper for backward compatibility, a
`TrainerFactory`/registry for mapping dataset names to data/model classes,
and a convenience `train(...)` entry point that selects the right classes
based on a dataset name.
"""

import warnings
warnings.filterwarnings("ignore", message=".*libjpeg.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*", category=UserWarning)

from abc import ABC, abstractmethod
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional, Callable

import torch
from torch import nn, optim

from .data import DATA_MODULES
from .models import MNISTNet, USPSNet, SVHNNet
from .constants import DATASET_STATS

# Define the project root relative to this file.
# This ensures paths are consistent regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("training")

class BaseTrainer(ABC):
    """Abstract base: subclasses implement :meth:`train` and return final metrics."""

    @abstractmethod
    def train(self):
        """Train a model and return training metrics."""
        raise NotImplementedError


class Trainer(BaseTrainer):
    """Generic training loop wrapping a constructed model + data module.

    Auto-detects device (CUDA → MPS → CPU) when not specified. :meth:`train`
    runs the loop, saves a checkpoint, and returns final metrics.
    """

    def __init__(
        self,
        data_module,
        model,
        epochs: int = 1,
        lr: float = 3e-4,
        checkpoint_path: str = "weights/model.pth",
        device: str | None = None,
        loss_fn: Optional[Callable] = None,
    ):
        """Configure the training run.

        Parameters
        ----------
        data_module
            Source of train/val/test DataLoaders.
        model : torch.nn.Module
            Model to train (already constructed).
        epochs : int, default 1
            Number of epochs to run.
        lr : float, default 3e-4
            Adam learning rate.
        checkpoint_path : str, default "weights/model.pth"
            Where to save the trained weights, relative to repo root.
        device : str, optional
            ``"cpu"``, ``"cuda"``, or ``"mps"``. Auto-detected if ``None``
            (CUDA → MPS → CPU). On MPS, DataLoader workers are clamped to 0
            to avoid a process-exit hang.
        loss_fn : Callable, optional
            Defaults to :class:`torch.nn.CrossEntropyLoss`.
        """
        self.data_module = data_module
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_path = PROJECT_ROOT / checkpoint_path
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        # MPS multiprocessing + DataLoader workers causes the process to hang on exit
        if self.device.type == "mps":
            self.data_module.num_workers = 0
        # Allow customizing the loss function; default to CrossEntropyLoss
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def train(self):
        """Run the training loop and save a checkpoint.

        Returns
        -------
        dict
            Keys ``"loss"``, ``"accuracy"``, ``"val_loss"``, ``"val_accuracy"``
            (final epoch), ``"epochs"`` (count run), and ``"checkpoint_path"``.
        """
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = self.loss_fn
        final_loss = 0.0
        final_accuracy = 0.0
        final_val_loss = 0.0
        final_val_accuracy = 0.0

        logger.info("Training on %s for %d epoch(s)", self.device, self.epochs)

        train_loader = self.data_module.train_loader()
        val_loader = self.data_module.val_loader()

        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader, start=1):
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

                # DEBUG: Log every 100 batches to avoid overwhelming the output.
                if batch_idx % 100 == 0:
                    logger.debug(
                        "epoch %d/%d  batch %d  loss=%.4f",
                        epoch, self.epochs, batch_idx, loss.item(),
                    )

            final_loss = running_loss / total if total else 0.0
            final_accuracy = correct / total if total else 0.0
            final_val_loss, final_val_accuracy = self._evaluate_loader(val_loader)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch, self.epochs, final_loss, final_accuracy, final_val_loss, final_val_accuracy,
            )

        del train_loader, val_loader

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)

        logger.info(
            "Done!  final_loss=%.4f  final_acc=%.4f  saved=%s",
            final_loss, final_accuracy, self.checkpoint_path,
        )

        return {
            "loss": final_loss,
            "accuracy": final_accuracy,
            "val_loss": final_val_loss,
            "val_accuracy": final_val_accuracy,
            "epochs": self.epochs,
            "checkpoint_path": str(self.checkpoint_path),
        }

    def _evaluate_loader(self, dataloader):
        """Compute average loss and accuracy on ``dataloader`` under ``torch.no_grad``."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        try:
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)

                    batch_size = labels.size(0)
                    running_loss += loss.item() * batch_size
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += batch_size
        finally:
            self.model.train()

        return (
            running_loss / total if total else 0.0,
            correct / total if total else 0.0,
        )


@dataclass
class DatasetSpec:
    """Registry entry binding a dataset name to its data module, model, checkpoint path, and image stats."""

    data_module_name: str
    model_cls: Type
    default_checkpoint: str
    image_size: int | tuple[int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    grayscale: bool
    trainer_cls: Type = Trainer


# Registry mapping dataset name -> DatasetSpec. Add SVHN/USPS entries as they
# become available in `src.data` and `src.models`.
DATASET_REGISTRY = {
    "mnist": DatasetSpec("mnist", MNISTNet, "weights/mnist.pth", **DATASET_STATS["mnist"]),
    "usps":  DatasetSpec("usps", USPSNet, "weights/usps.pth", **DATASET_STATS["usps"]),
    "svhn":  DatasetSpec("svhn", SVHNNet, "weights/svhn.pth", **DATASET_STATS["svhn"]),
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
        """Build a :class:`Trainer` for ``dataset_name`` using :data:`DATASET_REGISTRY` defaults.

        Parameters
        ----------
        dataset_name : str
            Key in :data:`DATASET_REGISTRY` (e.g. ``"mnist"``, ``"usps"``, ``"svhn"``).
        **kwargs
            Trainer overrides: ``epochs``, ``lr``, ``checkpoint_path``,
            ``device``, ``loss_fn``.

            Data-module overrides: ``data_dir``, ``batch_size``, ``num_workers``,
            ``image_size``, ``mean``, ``std``, ``grayscale``.

            Any unrecognized kwarg is forwarded to the data module.

        Returns
        -------
        Trainer
            Trainer instance with model, data module, and hyperparameters wired up.
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        spec = DATASET_REGISTRY[dataset_name]

        data_module_cls = DATA_MODULES[spec.data_module_name]
        
        # Collect data-related arguments. We pass everything in kwargs
        # so that different data modules can extract what they need.
        data_args = {
            "data_dir": kwargs.get("data_dir", "datasets"),
            "batch_size": kwargs.get("batch_size", 64),
            "num_workers": kwargs.get("num_workers", 2),
            "image_size": kwargs.get("image_size", spec.image_size),
            "mean": kwargs.get("mean", spec.mean),
            "std": kwargs.get("std", spec.std),
            "grayscale": kwargs.get("grayscale", spec.grayscale),
            **{k: v for k, v in kwargs.items() if k not in [
                "epochs", "lr", "checkpoint_path", "device", "loss_fn",
                "image_size", "mean", "std", "grayscale",
                "data_dir", "batch_size", "num_workers"
            ]}
        }
        data_module = data_module_cls(**data_args)

        # Resolve input_size for model (extract int from tuple if necessary)
        img_size = kwargs.get("image_size", spec.image_size)
        input_size = img_size[0] if isinstance(img_size, tuple) else img_size
        model = spec.model_cls(input_size=input_size)

        trainer_cls = spec.trainer_cls or Trainer
        checkpoint = kwargs.get("checkpoint_path") or spec.default_checkpoint

        return trainer_cls(
            data_module=data_module,
            model=model,
            epochs=kwargs.get("epochs", 10),
            lr=kwargs.get("lr", 1e-3),
            checkpoint_path=checkpoint,
            device=kwargs.get("device"),
            loss_fn=kwargs.get("loss_fn"),
        )


def train(dataset: str = "mnist", **kwargs):
    """Convenience entry point: select dataset, build trainer, and run.

    Parameters
    ----------
    dataset : str, default "mnist"
        Key in :data:`DATASET_REGISTRY` (e.g. ``"mnist"``, ``"usps"``, ``"svhn"``).
    **kwargs
        Forwarded to :meth:`TrainerFactory.create` — see that method for the
        full set of trainer and data-module overrides.

    Returns
    -------
    dict
        Same shape as :meth:`Trainer.train` — final loss/accuracy plus
        ``"epochs"`` and ``"checkpoint_path"``.
    """
    trainer = TrainerFactory.create(dataset, **kwargs)
    return trainer.train()


def _load_cfg(path: str) -> dict:
    """Parse a shell-style key=value config file into a dict of strings."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            cfg[key.strip()] = value.strip()
    return cfg


def build_arg_parser():
    """Build the command-line argument parser for training."""
    parser = argparse.ArgumentParser(description="Train a dataset model.")
    parser.add_argument(
        "--config", "-c",
        default=None,
        metavar="FILE",
        help="Path to a .cfg file (e.g. configs/mnist.cfg). Values act as defaults and are overridden by any explicit CLI flag.",
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Dataset to train on.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
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
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser


def main(argv=None):
    """Command-line entry point for training."""
    parser = build_arg_parser()

    # Pre-pass: find --config, load it, then set those values as defaults so
    # any explicit CLI flag still wins (CLI > config file > argparse defaults).
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config:
        cfg = _load_cfg(pre_args.config)
        action_map = {a.dest: a for a in parser._actions}
        typed_cfg = {
            key: (action_map[key].type(value) if action_map[key].type else value)
            for key, value in cfg.items()
            if key in action_map
        }
        parser.set_defaults(**typed_cfg)

    args = parser.parse_args(argv)

    from .utils import setup_logging
    setup_logging(args.log_level)

    logger.info(
        "Start:  dataset=%s epochs=%d batch_size=%d lr=%s device=%s",
        args.dataset, args.epochs, args.batch_size, args.lr, args.device or "auto",
    )

    # Map CLI loss name to a loss instance (or None to use trainer default)
    loss_cls_name = LOSS_MAP.get(args.loss)
    loss_fn = None
    if loss_cls_name is not None:
        loss_cls = getattr(nn, loss_cls_name)
        loss_fn = loss_cls()

    train(
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BaseTrainer",
    "Trainer",
    "DatasetSpec",
    "TrainerFactory",
    "build_arg_parser",
    "main",
    "train",
]
