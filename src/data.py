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

    def __init__(self, dataset_cls, mean, std, data_dir="datasets", batch_size=64, num_workers=2, download=True, **kwargs):
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
        self.dataset_kwargs = kwargs

    def _dataset(self, train):
        return self.dataset_cls(
            root=str(self.data_dir),
            train=train,
            download=self.download,
            transform=self.transform,
            **self.dataset_kwargs,
        )

    def train_loader(self):
        return DataLoader(
            self._dataset(train=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_loader(self):
        return DataLoader(
            self._dataset(train=False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MNISTDataModule(TorchvisionDataModule):
    def __init__(self, mean=0.1307, std=0.3081, data_dir="datasets", batch_size=64, num_workers=2, download=True, **kwargs):
        super().__init__(
            dataset_cls=datasets.MNIST,
            mean=mean,
            std=std,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            **kwargs,
        )

class USPSDataModule(TorchvisionDataModule):
    """USPS data module."""

    def __init__(self, mean=0.2471, std=0.2994, data_dir="datasets", batch_size=64, num_workers=2, download=True, **kwargs):
        super().__init__(
            dataset_cls=datasets.USPS,
            mean=mean,
            std=std,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            **kwargs,
        )


class SVHNDataModule(TorchvisionDataModule):
    """
    Data module for the SVHN dataset.

    This module handles downloading, preprocessing, and loading the SVHN
    training and test datasets. Images are converted to tensors and
    normalized using the standard SVHN channel-wise mean and standard
    deviation values.
    """
    
    MEAN = (0.4377, 0.4438, 0.4728)
    STD = (0.1980, 0.2010, 0.1970)

    def __init__(self, mean=None, std=None, data_dir="datasets", batch_size=64, num_workers=2, download=True, **kwargs):
        """ Initialize the SVHN data module. """
        
        BaseDataModule.__init__(self)
        self.dataset_cls = datasets.SVHN
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )
        self.dataset_kwargs = kwargs

    def _dataset(self, train):
        """
        Create an SVHN dataset instance.

        Parameters
        ----------
        train : bool
            If True, return the training split; otherwise, return the test
            split.

        Returns
        -------
        torchvision.datasets.SVHN
            Configured SVHN dataset instance.
        """
        
        return self.dataset_cls(
            root=str(self.data_dir),
            split="train" if train else "test",
            download=self.download,
            transform=self.transform,
            **self.dataset_kwargs,
        )


DATA_MODULES = {
    "mnist": MNISTDataModule,
    "usps": USPSDataModule,
    "svhn": SVHNDataModule,
}


def get_loaders(dataset="mnist", mean=0.1307, std=0.3081, data_dir="datasets", batch_size=64, num_workers=2, download=True, **kwarg):
    """Convenience entry point: get train and test loaders for a registered dataset.

    Example: `get_loaders(dataset="mnist", batch_size=32)`
    """
    if dataset not in DATA_MODULES:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATA_MODULES.keys())}")

    data_module_cls = DATA_MODULES[dataset]
    data_module = data_module_cls(
        mean=mean,
        std=std,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        **kwarg
    )
    return data_module.train_loader(), data_module.test_loader()


__all__ = ["BaseDataModule", "TorchvisionDataModule", "MNISTDataModule", "USPSDataModule", "SVHNDataModule", "get_loaders", "DATA_MODULES"]
