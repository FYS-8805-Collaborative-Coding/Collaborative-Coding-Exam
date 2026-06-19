"""Dataset loaders and data-module base classes."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.constants import DATASET_STATS
random_split = getattr(torch.utils.data, "random_split", None)


def _to_rgb(img):
    """Convert a PIL image to RGB.

    Defined at module level (not as a lambda) so it is picklable — DataLoader
    workers on Python 3.14 pickle the dataset's transform when spawning.
    """
    return img.convert("RGB")


class BaseDataModule(ABC):
    """Abstract base: subclasses provide :meth:`train_loader` and :meth:`test_loader`."""

    @abstractmethod
    def train_loader(self):
        """Return the training data loader."""
        raise NotImplementedError

    @abstractmethod
    def test_loader(self):
        """Return the test data loader."""
        raise NotImplementedError


class TorchvisionDataModule(BaseDataModule):
    """Generic data module wrapping a torchvision dataset class.

    Builds the resize → channel → tensor → normalize transform pipeline,
    splits the training set into train/val (seeded), and exposes train/val/test
    DataLoaders. Subclasses (e.g. :class:`MNISTDataModule`) just supply the
    dataset class and its normalization stats.
    """

    def __init__(
        self, 
        dataset_cls, 
        mean, 
        std, 
        image_size: int | tuple[int, int] = 28, 
        grayscale: bool = True, 
        data_dir="datasets", 
        batch_size=64, 
        num_workers=2, 
        download=True,
        val_split=0.1,
        val_seed=42,
        **kwargs
    ):
        """Wire up the torchvision dataset class with its normalization, transforms, and DataLoader config."""
        self.dataset_cls = dataset_cls
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

        norm_mean = mean if isinstance(mean, (list, tuple)) else (mean,)
        norm_std = std if isinstance(std, (list, tuple)) else (std,)

        self.val_split = val_split
        self.val_seed = val_seed
        self._train_val_split = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size if isinstance(image_size, tuple) else (image_size, image_size)),
                transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(_to_rgb),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
        self.dataset_kwargs = kwargs

    def _dataset(self, train):
        """Instantiate the underlying torchvision dataset for the requested split."""
        ds_kwargs = self.dataset_kwargs.copy()
        if self.dataset_cls == datasets.SVHN:
            ds_kwargs["split"] = "train" if train else "test"
        else:
            ds_kwargs["train"] = train

        return self.dataset_cls(
            root=str(self.data_dir),
            download=self.download,
            transform=self.transform,
            **ds_kwargs,
        )

    def _train_val_datasets(self):
        """Return cached ``(train, val)`` subsets of the training set, split once with ``val_seed``."""
        if self._train_val_split is None:
            full_train = self._dataset(train=True)
            if random_split is None:
                self._train_val_split = (full_train, full_train)
            else:
                val_size = int(len(full_train) * self.val_split)
                train_size = len(full_train) - val_size
                generator = torch.Generator().manual_seed(self.val_seed)
                self._train_val_split = random_split(
                    full_train,
                    [train_size, val_size],
                    generator=generator,
                )
        return self._train_val_split

    def train_loader(self):
        """Return a shuffling DataLoader over the training split (val subset excluded)."""
        train_dataset, _ = self._train_val_datasets()
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_loader(self):
        """Return a non-shuffling DataLoader over the held-out validation subset."""
        _, val_dataset = self._train_val_datasets()
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_loader(self):
        """Return a non-shuffling DataLoader over the test split."""
        return DataLoader(
            self._dataset(train=False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MNISTDataModule(TorchvisionDataModule):
    """Data module for the MNIST handwritten-digit dataset."""

    def __init__(self, mean=None, std=None, data_dir="datasets", batch_size=64, num_workers=2, download=True, image_size=None, grayscale=None, **kwargs):
        """Configure the shared :class:`TorchvisionDataModule` with MNIST defaults."""
        stats = DATASET_STATS["mnist"]
        super().__init__(
            dataset_cls=datasets.MNIST,
            mean=mean or stats["mean"],
            std=std or stats["std"],
            image_size=image_size if image_size is not None else stats["image_size"],
            grayscale=grayscale if grayscale is not None else stats["grayscale"],
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            **kwargs,
        )

class USPSDataModule(TorchvisionDataModule):
    """Data module for the USPS handwritten-digit dataset.

    Downloads land under ``<data_dir>/USPS/``. Defaults to USPS's
    standard normalization (``mean=0.2471``, ``std=0.2994``).
    """
    def __init__(self, mean=None, std=None, data_dir="datasets", batch_size=64, num_workers=2, download=True, image_size=None, grayscale=None, **kwargs):
        """Configure the shared :class:`TorchvisionDataModule` with USPS defaults."""
        stats = DATASET_STATS["usps"]
        super().__init__(
            dataset_cls=datasets.USPS,
            mean=mean or stats["mean"],
            std=std or stats["std"],
            image_size=image_size if image_size is not None else stats["image_size"],
            grayscale=grayscale if grayscale is not None else stats["grayscale"],
            data_dir=str(Path(data_dir) / "USPS"),
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
    def __init__(self, mean=None, std=None, data_dir="datasets", batch_size=64, num_workers=2, download=True, image_size=None, grayscale=None, **kwargs):
        """Configure the shared :class:`TorchvisionDataModule` with SVHN defaults."""
        stats = DATASET_STATS["svhn"]
        super().__init__(
            dataset_cls=datasets.SVHN,
            mean=mean or stats["mean"],
            std=std or stats["std"],
            image_size=image_size if image_size is not None else stats["image_size"],
            grayscale=grayscale if grayscale is not None else stats["grayscale"],
            val_split=kwargs.pop("val_split", 0.1), # Default to 0.1 if not provided
            val_seed=kwargs.pop("val_seed", 42), # Default to 42 if not provided
            data_dir=str(Path(data_dir) / "SVHN"),
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            **kwargs,
        )
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


def get_loaders(dataset="mnist", mean=None, std=None, data_dir="datasets", batch_size=64, num_workers=2, download=True, image_size=None, grayscale=None, **kwarg):
    """Convenience entry point: get train and test loaders for a registered dataset.

    Example: `get_loaders(dataset="mnist", batch_size=32)`
    """
    if dataset not in DATA_MODULES:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATA_MODULES.keys())}")
    
    stats = DATASET_STATS[dataset]

    data_module_cls = DATA_MODULES[dataset]
    data_module = data_module_cls(
        mean=mean or stats["mean"],
        std=std or stats["std"],
        image_size=image_size if image_size is not None else stats["image_size"],
        grayscale=grayscale if grayscale is not None else stats["grayscale"],
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        **kwarg
    )
    return data_module.train_loader(), data_module.test_loader()


__all__ = ["BaseDataModule", "TorchvisionDataModule", "MNISTDataModule", "USPSDataModule", "SVHNDataModule", "get_loaders", "DATA_MODULES"]
