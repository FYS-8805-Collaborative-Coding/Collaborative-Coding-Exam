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


class TorchvisionDataModule(BaseDataModule):
    """Generic data module for standard torchvision datasets."""

    def __init__(self, dataset_cls, mean, std, data_dir="datasets", batch_size=64, num_workers=2, download=True):
        self.dataset_cls = dataset_cls
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        )

    def _dataset(self, train):
        return self.dataset_cls(
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


class MNISTDataModule(TorchvisionDataModule):
    """MNIST data module."""

    def __init__(self, data_dir="datasets", batch_size=64, num_workers=2, download=True):
        super().__init__(
            dataset_cls=datasets.MNIST,
            mean=0.1307,
            std=0.3081,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
        )


class USPSDataModule(TorchvisionDataModule):
    """USPS data module."""

    def __init__(self, data_dir="datasets", batch_size=64, num_workers=2, download=True):
        super().__init__(
            dataset_cls=datasets.USPS,
            mean=0.2471,
            std=0.2994,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
        )


DATA_MODULES = {
    "mnist": MNISTDataModule,
    "usps": USPSDataModule,
}


def get_loaders(dataset="mnist", data_dir="datasets", batch_size=64, num_workers=2, download=True):
    """Generic helper to get train and test loaders for any registered dataset."""
    if dataset not in DATA_MODULES:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATA_MODULES.keys())}")

    data_module_cls = DATA_MODULES[dataset]
    data_module = data_module_cls(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
    )
    return data_module.train_loader(), data_module.test_loader()


__all__ = ["BaseDataModule", "TorchvisionDataModule", "MNISTDataModule", "USPSDataModule", "get_loaders", "DATA_MODULES"]
