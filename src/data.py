"""Dataset loaders and data-module base classes."""

from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BaseDataModule(ABC):
    """Base class for dataset-specific data loaders."""

    @abstractmethod
    def train_loader(self):
        """Return the training data loader."""
        raise NotImplementedError

    @abstractmethod
    def test_loader(self):
        """Return the test data loader."""
        raise NotImplementedError


class MNISTDataModule(BaseDataModule):
    """MNIST data module."""

    def __init__(self, data_dir="datasets", batch_size=64, num_workers=2, download=True):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def _dataset(self, train):
        return datasets.MNIST(
            root=str(self.data_dir),
            train=train,
            download=self.download,
            transform=self.transform,
        )

    def train_loader(self):
        return DataLoader(
            self._dataset(train=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_loader(self):
        return DataLoader(
            self._dataset(train=False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def get_mnist_loaders(data_dir="datasets", batch_size=64, num_workers=2, download=True):
    data_module = MNISTDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
    )
    return data_module.train_loader(), data_module.test_loader()


__all__ = ["BaseDataModule", "MNISTDataModule", "get_mnist_loaders"]
