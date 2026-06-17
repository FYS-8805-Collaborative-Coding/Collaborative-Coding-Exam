import importlib.util
import sys
import types
from pathlib import Path

import pytest

"""Unit tests for src/training.py.

These tests inject lightweight fake modules into `sys.modules` so the
training module can be imported and exercised without PyTorch, GPUs,
or real datasets. Tests focus on CLI parsing, TrainerFactory wiring,
and delegation behavior rather than numeric training correctness.
"""


class DummyDataModule:
    # Lightweight stand-ins used across the tests so we don't need real
    # datasets, DataLoaders, or model classes.

    def __init__(self, data_dir="datasets", batch_size=64, num_workers=2, **kwargs):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def val_loader(self):
        return []

    def train_loader(self):
        return []


class DummyModel:
    # Accept input_size and other kwargs passed by TrainerFactory
    def __init__(self, **kwargs):
        self.created = True


class DummyTrainer:
    def __init__(self, data_module, model, epochs=1, lr=1e-3, checkpoint_path="weights/model.pth", device=None, loss_fn=None):
        self.data_module = data_module
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_path = Path(__file__).resolve().parents[1] / checkpoint_path
        self.device = device
        self.loss_fn = None

    def train(self):
        return {
            "loss": 0.0,
            "accuracy": 1.0,
            "epochs": self.epochs,
            "checkpoint_path": self.checkpoint_path,
        }

            # Build small fake modules and temporarily install them into
            # `sys.modules`. This lets `src/training.py` import `torch`,
            # `data`, and `models` without needing the real libraries.

class DummySpec:
    def __init__(self):
        self.data_module_name = "dummy"
        self.model_cls = DummyModel
        self.default_checkpoint = "weights/dummy.pth"
        self.image_size = 28
        self.mean = (0.5,)
        self.std = (0.5,)
        self.grayscale = True
        self.trainer_cls = DummyTrainer


class FakeCrossEntropyLoss:
    def __call__(self, logits, labels):
        return self

    def backward(self):
        return None


class FakeAdam:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        return None

    def step(self):
        return None


# Build small fake modules and temporarily install them into
# `sys.modules`. This lets `src/training.py` import `torch`,
# `data`, and `models` without needing the real libraries.

def load_training_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda value: value
    fake_torch.save = lambda *args, **kwargs: None
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    # Mock torch.ops.torchvision._cuda_version for torchvision import
    fake_torch.ops = types.SimpleNamespace(
        load_library=lambda x: None,
        torchvision=types.SimpleNamespace(_cuda_version=lambda: -1)) # Mock torch.ops.torchvision._cuda_version
    fake_nn = types.ModuleType("torch.nn")

    class FakeModule:
        def parameters(self):
            return []

        def to(self, device): # Ensure the factory uses the registry mapping to construct trainer objects
            return self

        def train(self):
            return self

        def state_dict(self):
            return {}

    fake_nn.Module = FakeModule
    fake_nn.CrossEntropyLoss = FakeCrossEntropyLoss

    fake_optim = types.ModuleType("torch.optim")
    fake_optim.Adam = FakeAdam

    fake_torch.nn = fake_nn
    fake_torch.optim = fake_optim

    original_modules = {
        "torch": sys.modules.get("torch"),
        "torch.nn": sys.modules.get("torch.nn"),
        "torch.optim": sys.modules.get("torch.optim"),
        "src.data": sys.modules.get("src.data"),
        "src.models": sys.modules.get("src.models"),
        "src.constants": sys.modules.get("src.constants"),
    }

    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn
    sys.modules["torch.optim"] = fake_optim

    sys.modules["src.data"] = types.SimpleNamespace(
        DATA_MODULES={"mnist": DummyDataModule, "usps": DummyDataModule, "svhn": DummyDataModule}
    )
    sys.modules["src.models"] = types.SimpleNamespace(
        MNISTNet=DummyModel, USPSNet=DummyModel, SVHNNet=DummyModel
    )
    sys.modules["src.constants"] = types.SimpleNamespace(
        DATASET_STATS={
            "mnist": {"image_size": 28, "mean": (0.1307,), "std": (0.3081,), "grayscale": True},
            "usps":  {"image_size": 16, "mean": (0.2471,), "std": (0.2994,), "grayscale": True},
            "svhn":  {"image_size": 32, "mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970), "grayscale": False},
        }
    )

    module_path = Path(__file__).resolve().parents[1] / "src" / "training.py"
    spec = importlib.util.spec_from_file_location("src.training_under_test", module_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return module


@pytest.fixture(scope="module")
def training():
    return load_training_module()


def test_build_arg_parser_defaults(training):
    parser = training.build_arg_parser()
    args = parser.parse_args([])

    assert args.dataset == "mnist"
    assert args.epochs == 1
    assert args.lr == pytest.approx(3e-4)
    assert args.batch_size == 64
    assert args.checkpoint_path is None
    assert args.data_dir == "datasets"
    assert args.num_workers == 2
    assert args.device is None



def test_trainer_factory_uses_registry(training, monkeypatch):
    monkeypatch.setitem(training.DATASET_REGISTRY, "dummy", DummySpec())
    monkeypatch.setitem(training.DATA_MODULES, "dummy", DummyDataModule)

    trainer = training.TrainerFactory.create(
        "dummy",
        epochs=3,
        lr=0.01,
        batch_size=8,
        data_dir="sample-data",
        num_workers=0,
        checkpoint_path="weights/custom.pth",
        device="cpu",
    )

    assert isinstance(trainer, DummyTrainer)
    assert trainer.epochs == 3
    assert trainer.lr == pytest.approx(0.01) #
    assert Path(trainer.checkpoint_path) == training.PROJECT_ROOT / "weights/custom.pth" #
    assert trainer.device == "cpu"
    assert trainer.data_module.data_dir == "sample-data"
    assert trainer.data_module.batch_size == 8
    assert trainer.data_module.num_workers == 0
    assert isinstance(trainer.model, DummyModel)


def test_train_delegates_to_factory_and_trainer(training, monkeypatch):
    captured = {}

    class FakeFactory:
        @classmethod
        def create(cls, dataset_name, **kwargs):
            captured["dataset_name"] = dataset_name
            captured["kwargs"] = kwargs

            class FakeTrainer:
                def train(self):
                    return {"status": "ok"}

            return FakeTrainer()

    monkeypatch.setattr(training, "TrainerFactory", FakeFactory)

    result = training.train(dataset="mnist", epochs=5, batch_size=32, device="cpu")

    assert result == {"status": "ok"}
    assert captured["dataset_name"] == "mnist"
    assert captured["kwargs"]["epochs"] == 5
    assert captured["kwargs"]["batch_size"] == 32
    assert captured["kwargs"]["device"] == "cpu"


def test_parser_rejects_unknown_dataset(training):
    parser = training.build_arg_parser()

    with pytest.raises(SystemExit):
        # argparse exits with SystemExit when a choice is invalid
        parser.parse_args(["--dataset", "not-a-dataset"])


def test_parser_rejects_unknown_argument(training):
    parser = training.build_arg_parser()

    with pytest.raises(SystemExit):
        # argparse exits when an unknown argument is passed
        parser.parse_args(["--this-flag-does-not-exist"])
