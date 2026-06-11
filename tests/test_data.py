"""Tests for src/data.py — no real datasets, torch, or network access needed."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_data_module():
    """Load src/data.py with lightweight fakes in place of torch and torchvision."""

    class FakeNormalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

    class FakeCompose:
        def __init__(self, transform_list):
            self.transforms = transform_list

    fake_transforms = types.ModuleType("torchvision.transforms")
    fake_transforms.Compose = FakeCompose
    fake_transforms.ToTensor = object
    fake_transforms.Normalize = FakeNormalize

    class FakeDataset:
        def __init__(self, root, train=None, split=None, download=True, transform=None, **_):
            self.root = root
            self.train = train
            self.split = split
            self.download = download
            self.transform = transform

    class FakeMNIST(FakeDataset):
        pass

    class FakeUSPS(FakeDataset):
        pass

    class FakeSVHN(FakeDataset):
        pass

    fake_datasets = types.ModuleType("torchvision.datasets")
    fake_datasets.MNIST = FakeMNIST
    fake_datasets.USPS = FakeUSPS
    fake_datasets.SVHN = FakeSVHN

    class FakeDataLoader:
        def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.drop_last = drop_last

    fake_utils_data = types.ModuleType("torch.utils.data")
    fake_utils_data.DataLoader = FakeDataLoader

    fake_utils = types.ModuleType("torch.utils")
    fake_utils.data = fake_utils_data

    fake_torch = types.ModuleType("torch")
    fake_torch.utils = fake_utils

    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.datasets = fake_datasets
    fake_torchvision.transforms = fake_transforms

    to_inject = {
        "torch": fake_torch,
        "torch.utils": fake_utils,
        "torch.utils.data": fake_utils_data,
        "torchvision": fake_torchvision,
        "torchvision.datasets": fake_datasets,
        "torchvision.transforms": fake_transforms,
    }
    originals = {k: sys.modules.get(k) for k in to_inject}
    for k, v in to_inject.items():
        sys.modules[k] = v

    module_path = Path(__file__).resolve().parents[1] / "src" / "data.py"
    spec = importlib.util.spec_from_file_location("data_under_test", module_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    finally:
        for k, original in originals.items():
            if original is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = original

    return module, fake_datasets


def _normalize(data_module):
    return data_module.transform.transforms[-1]


# ─── registry ─────────────────────────────────────────────────────────────────

def test_data_modules_registry_has_expected_keys():
    module, _ = _load_data_module()
    assert set(module.DATA_MODULES.keys()) == {"mnist", "usps", "svhn"}


def test_base_data_module_is_abstract():
    module, _ = _load_data_module()
    with pytest.raises(TypeError):
        module.BaseDataModule()


# ─── get_loaders ──────────────────────────────────────────────────────────────

def test_get_loaders_unknown_dataset_raises_value_error():
    module, _ = _load_data_module()
    with pytest.raises(ValueError, match="Unknown dataset"):
        module.get_loaders(dataset="imagenet")


@pytest.mark.parametrize("name", ["mnist", "usps", "svhn"])
def test_get_loaders_known_datasets_return_two_loaders(name):
    module, _ = _load_data_module()
    train_loader, test_loader = module.get_loaders(dataset=name)
    assert train_loader is not None
    assert test_loader is not None


def test_get_loaders_passes_batch_size():
    module, _ = _load_data_module()
    train_loader, test_loader = module.get_loaders(dataset="mnist", batch_size=128)
    assert train_loader.batch_size == 128
    assert test_loader.batch_size == 128


# ─── TorchvisionDataModule constructor ────────────────────────────────────────

def test_torchvision_data_module_stores_constructor_params():
    module, fake_datasets = _load_data_module()
    dm = module.TorchvisionDataModule(
        dataset_cls=fake_datasets.MNIST,
        mean=0.5,
        std=0.25,
        data_dir="custom/path",
        batch_size=32,
        num_workers=4,
        download=False,
    )
    assert dm.batch_size == 32
    assert dm.num_workers == 4
    assert dm.download is False
    assert dm.data_dir == Path("custom/path")


def test_torchvision_data_module_transform_uses_mean_std():
    module, fake_datasets = _load_data_module()
    dm = module.TorchvisionDataModule(dataset_cls=fake_datasets.MNIST, mean=0.42, std=0.11)
    n = _normalize(dm)
    assert n.mean == (0.42,)
    assert n.std == (0.11,)


# ─── dataset-specific defaults ────────────────────────────────────────────────

def test_mnist_default_normalization():
    module, _ = _load_data_module()
    n = _normalize(module.MNISTDataModule())
    assert n.mean == pytest.approx((0.1307,))
    assert n.std == pytest.approx((0.3081,))


def test_usps_default_normalization():
    module, _ = _load_data_module()
    n = _normalize(module.USPSDataModule())
    assert n.mean == pytest.approx((0.2471,))
    assert n.std == pytest.approx((0.2994,))


def test_svhn_uses_class_level_mean_std_constants():
    module, _ = _load_data_module()
    dm = module.SVHNDataModule()
    n = _normalize(dm)
    assert n.mean == pytest.approx((0.4377, 0.4438, 0.4728))
    assert n.std == pytest.approx((0.1980, 0.2010, 0.1970))


# ─── _dataset split/train kwarg routing ───────────────────────────────────────

def test_torchvision_dataset_passes_train_bool():
    module, _ = _load_data_module()
    dm = module.MNISTDataModule()
    assert dm._dataset(train=True).train is True
    assert dm._dataset(train=False).train is False


def test_svhn_dataset_passes_split_string_not_train():
    module, _ = _load_data_module()
    dm = module.SVHNDataModule()
    train_ds = dm._dataset(train=True)
    test_ds = dm._dataset(train=False)
    assert train_ds.split == "train"
    assert test_ds.split == "test"
    assert train_ds.train is None


# ─── DataLoader flags ─────────────────────────────────────────────────────────

def test_train_loader_is_shuffled_and_drops_last():
    module, _ = _load_data_module()
    loader = module.MNISTDataModule(batch_size=16).train_loader()
    assert loader.shuffle is True
    assert loader.drop_last is True
    assert loader.batch_size == 16


def test_test_loader_is_not_shuffled():
    module, _ = _load_data_module()
    loader = module.MNISTDataModule(batch_size=16).test_loader()
    assert loader.shuffle is False
    assert loader.batch_size == 16
