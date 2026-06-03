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

    def train_loader(self):
        return []


class DummyModel:
    def __init__(self):
        self.created = True


class DummyTrainer:
    def __init__(self, data_module, model, epochs=1, lr=1e-3, checkpoint_path="weights/model.pth", device=None, loss_fn=None):
        self.data_module = data_module
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_path = checkpoint_path
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


def load_training_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda value: value
    fake_torch.save = lambda *args, **kwargs: None
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_nn = types.ModuleType("torch.nn")

    class FakeModule:
        def parameters(self):
            return []

        def to(self, device):


# Ensure the factory uses the registry mapping to construct trainer objects
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

    fake_data = types.ModuleType("data")
    fake_data.DATA_MODULES = {"mnist": DummyDataModule, "usps": DummyDataModule}

    fake_models = types.ModuleType("models")
    fake_models.MNISTNet = DummyModel
    fake_models.USPSNet = DummyModel

    original_modules = {


# Verify that `train()` delegates to TrainerFactory.create and calls
# the trainer's `train()` method, returning its result.
        "torch": sys.modules.get("torch"),
        "torch.nn": sys.modules.get("torch.nn"),
        "torch.optim": sys.modules.get("torch.optim"),
        "data": sys.modules.get("data"),
        "models": sys.modules.get("models"),
    }

    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn
    sys.modules["torch.optim"] = fake_optim
    sys.modules["data"] = fake_data
    sys.modules["models"] = fake_models

    module_path = Path(__file__).resolve().parents[1] / "src" / "training.py"
    spec = importlib.util.spec_from_file_location("training_under_test", module_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


# argparse should raise SystemExit for an unknown --dataset choice
    return module



def test_build_arg_parser_defaults():
    training = load_training_module()
    parser = training.build_arg_parser()
    args = parser.parse_args([])

    assert args.dataset == "mnist"
    assert args.epochs == 1
    assert args.lr == pytest.approx(1e-3)
    assert args.batch_size == 64
    assert args.checkpoint_path is None
    assert args.data_dir == "datasets"
    assert args.num_workers == 2
    assert args.device is None



def test_trainer_factory_uses_registry(monkeypatch):
    training = load_training_module()
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
    assert trainer.lr == pytest.approx(0.01)
    assert trainer.checkpoint_path == "weights/custom.pth"
    assert trainer.device == "cpu"
    assert trainer.data_module.data_dir == "sample-data"
    assert trainer.data_module.batch_size == 8
    assert trainer.data_module.num_workers == 0
    assert isinstance(trainer.model, DummyModel)



def test_train_delegates_to_factory_and_trainer(monkeypatch):
    training = load_training_module()
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


def test_parser_rejects_unknown_dataset():
    training = load_training_module()
    parser = training.build_arg_parser()

    with pytest.raises(SystemExit):
        # argparse exits with SystemExit when a choice is invalid
        parser.parse_args(["--dataset", "not-a-dataset"])


def test_parser_rejects_unknown_argument():
    training = load_training_module()
    parser = training.build_arg_parser()

    with pytest.raises(SystemExit):
        # argparse exits when an unknown argument is passed
        parser.parse_args(["--this-flag-does-not-exist"])
