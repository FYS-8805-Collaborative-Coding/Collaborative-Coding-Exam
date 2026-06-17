"""
Unit tests for the inference pipeline (src/inference.py).

These tests verify:
1. File discovery: Correctly identifying and filtering image files in directories.
2. Path resolution: Ensuring model weights are found relative to the project root.
3. Transformation: Confirming image preprocessing produces correct tensor dimensions.
4. Factory/Registry: Validating that model aliases map to correct specifications.
5. Core Prediction: Verifying the flow from image input to final class index via mocked models.
6. Isolation: Uses a comprehensive mocking strategy to run without real Torch/NumPy binaries.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import types
import importlib.util

# Mock PIL if not available to support isolated testing in restricted environments.
try:
    from PIL import Image
except ImportError:
    # Create fake PIL and PIL.Image modules so that @patch("PIL.Image.open")
    # and top-level imports in src/inference.py do not raise ModuleNotFoundError.
    _pil = types.ModuleType("PIL")
    _pil_image = types.ModuleType("PIL.Image")
    _pil.Image = _pil_image
    sys.modules["PIL"] = _pil
    sys.modules["PIL.Image"] = _pil_image

    Image = MagicMock()
    _pil_image.Image = Image

    def _mock_new(mode, size):
        img = MagicMock()
        img.mode = mode
        img.width, img.height = (size, size) if isinstance(size, int) else size
        img.size = (img.width, img.height)
        img.resize = MagicMock(side_effect=lambda s: _mock_new(mode, s))
        img.convert = MagicMock(side_effect=lambda mod: _mock_new(mod, (img.width, img.height)))
        return img
    Image.new = _mock_new
    _pil_image.new = _mock_new
    _pil_image.open = MagicMock(side_effect=lambda p: _mock_new("RGB", (32, 32)))

# --- Start of mocking setup ---
class MockTensor:
    """A mock object that behaves like a torch.Tensor for basic operations."""
    def __init__(self, shape, value=0.0):
        self.shape = shape
        if isinstance(shape, int):
            self.shape = (shape,)
        if not self.shape:
            self._data = value
        elif len(self.shape) == 1:
            self._data = [value] * self.shape[0]
        elif len(self.shape) == 2:
            self._data = [[value] * self.shape[1] for _ in range(self.shape[0])]
        elif len(self.shape) == 3:
            self._data = [[[value] * self.shape[2] for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        elif len(self.shape) == 4:
            self._data = [[[[value] * self.shape[3] for _ in range(self.shape[2])] for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        else:
            self._data = value

    def _get_nested(self, keys):
        """Traverse self._data following a sequence of integer keys."""
        current = self._data
        for k in keys:
            current = current[k]
        return current

    def _set_nested(self, keys, value):
        """Traverse self._data following a sequence of integer keys and set the final element."""
        current = self._data
        for k in keys[:-1]:
            current = current[k]
        current[keys[-1]] = value

    def unsqueeze(self, dim):
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        return MockTensor(tuple(new_shape))

    def argmax(self, dim):
        if dim == 1 and len(self.shape) == 2:
            # Find the argmax across all batch items; return result for batch item 0
            max_val = -float('inf')
            max_idx = 0
            for i, val in enumerate(self._data[0]):
                if val > max_val:
                    max_val = val
                    max_idx = i
            return MockTensor((self.shape[0],), value=max_idx)
        return MockTensor((self.shape[0],), value=0)

    def item(self):
        if not self.shape:
            return self._data
        if len(self.shape) == 1 and self.shape[0] == 1:
            return self._data[0]
        if len(self.shape) == 2 and self.shape[0] == 1 and self.shape[1] == 1:
            return self._data[0][0]
        return self._data

    def to(self, device):
        return self

    @property
    def device(self):
        return "cpu"

    def __getitem__(self, key):
        if not self.shape:
            return self._data
        keys = list(key) if isinstance(key, tuple) else [key]
        return self._get_nested(keys)

    def __setitem__(self, key, value):
        """Correctly handle both single-key and tuple-key assignment."""
        if not self.shape:
            self._data = value
            return
        keys = list(key) if isinstance(key, tuple) else [key]
        self._set_nested(keys, value)

    def __len__(self):
        return self.shape[0] if self.shape else 0


class DummyModel(MagicMock):
    """A mock for torch.nn.Module."""
    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.to_called_with = device
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def parameters(self):
        return []


class FakeTorch:
    """A mock for the torch module."""
    def __init__(self):
        self.device = lambda value: value
        self.nn = types.SimpleNamespace(Module=DummyModel)
        self.Tensor = MockTensor
        self.no_grad_context = MagicMock()
        self.cuda = types.SimpleNamespace(is_available=lambda: False)

    def load(self, path, map_location=None):
        return {}

    def zeros(self, shape, **kwargs):
        return MockTensor(shape, value=0.0)

    def randn(self, *shape, **kwargs):
        return MockTensor(shape, value=0.1)

    def no_grad(self):
        return self.no_grad_context


def _make_fake_modules():
    """Build and return all fake modules needed to load inference without real torch."""
    fake_torch = FakeTorch()
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = DummyModel
    fake_torch.nn = fake_nn

    fake_models = types.ModuleType("src.models")
    fake_models.MNISTNet = DummyModel
    fake_models.USPSNet = DummyModel
    fake_models.SVHNNet = DummyModel

    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision_transforms = types.ModuleType("torchvision.transforms")
    fake_torchvision.transforms = fake_torchvision_transforms

    def mock_compose(transforms_list):
        def apply_transforms(img):
            for t in transforms_list:
                img = t(img)
            return img
        return apply_transforms

    fake_torchvision_transforms.Compose = mock_compose
    fake_torchvision_transforms.Resize = (
        lambda size: lambda img: img.resize(
            (size[1], size[0]) if isinstance(size, tuple) else (size, size)
        )
    )
    fake_torchvision_transforms.ToTensor = (
        lambda: lambda img: MockTensor((1, img.height, img.width) if img.mode == "L" else (3, img.height, img.width))
    )
    fake_torchvision_transforms.Normalize = lambda mean, std: lambda tensor: tensor
    fake_torchvision_transforms.Grayscale = lambda num_output_channels: lambda img: img.convert("L")
    fake_torchvision_transforms.Lambda = lambda f: f

    return {
        "torch": fake_torch,
        "torch.nn": fake_nn,
        "src.models": fake_models,
        "models": fake_models,
        "torchvision": fake_torchvision,
        "torchvision.transforms": fake_torchvision_transforms,
        "PIL": sys.modules.get("PIL"),
        "PIL.Image": sys.modules.get("PIL.Image"),
    }


def load_inference_module():
    """Load src.inference with mocked torch and model dependencies."""
    original_modules = {
        name: sys.modules.get(name)
        for name in [
            "torch", "torch.nn", "src.models", "models",
            "src.inference_under_test", "torchvision", "torchvision.transforms",
            "PIL", "PIL.Image",
        ]
    }

    fake_modules = _make_fake_modules()
    for name, mod in fake_modules.items():
        sys.modules[name] = mod

    module_path = Path(__file__).resolve().parents[1] / "src" / "inference.py"
    spec = importlib.util.spec_from_file_location("src.inference_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    # Register so @dataclass can resolve forward references in Python 3.13+.
    sys.modules["src.inference_under_test"] = module

    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return module


# ---------------------------------------------------------------------------
# Module-scoped fixture — loads the module once per test session rather than
# once per test, which is both faster and avoids repeated sys.modules churn.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def infer():
    return load_inference_module()


# --- Tests ---

def _save_image(path, size=(32, 32), mode="RGB"):
    """Write a real, decodable image to ``path`` (format inferred from suffix)."""
    Image.new(mode, size).save(path)


def test_iter_image_paths_directory(tmp_path, infer):
    """Iterator finds real images (by content) and returns them sorted."""
    _save_image(tmp_path / "img2.png")
    _save_image(tmp_path / "img1.jpg")
    (tmp_path / "notes.txt").write_text("data")  # not an image -> ignored

    paths = list(infer.iter_image_paths(tmp_path))

    assert len(paths) == 2
    assert paths[0].name == "img1.jpg"
    assert paths[1].name == "img2.png"


def test_iter_image_paths_file(tmp_path, infer):
    """Iterator works for a single valid image file."""
    img_path = tmp_path / "test.png"
    _save_image(img_path)

    paths = list(infer.iter_image_paths(img_path))
    assert paths == [img_path]


def test_iter_image_paths_accepts_extensionless_image(tmp_path, infer):
    """A real image with no file extension is accepted (detection is by content)."""
    img_path = tmp_path / "digit"  # no extension
    _save_image(img_path.with_suffix(".png"))
    (img_path.with_suffix(".png")).rename(img_path)

    paths = list(infer.iter_image_paths(img_path))
    assert paths == [img_path]


def test_iter_image_paths_rejects_non_image(tmp_path, infer):
    """A non-image file (e.g. .txt) is rejected, regardless of name."""
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("data")

    with pytest.raises(ValueError, match="Not a valid image file"):
        list(infer.iter_image_paths(txt_path))


def test_iter_image_paths_rejects_text_renamed_to_png(tmp_path, infer):
    """A text file masquerading as a .png is rejected (content is checked)."""
    fake = tmp_path / "fake.png"
    fake.write_text("this is not an image")

    with pytest.raises(ValueError, match="Not a valid image file"):
        list(infer.iter_image_paths(fake))


def test_iter_image_paths_empty_directory(tmp_path, infer):
    """A directory with no valid images raises ValueError."""
    with pytest.raises(ValueError, match="No valid image files found"):
        list(infer.iter_image_paths(tmp_path))


def test_resolve_checkpoint_path_relative(infer):
    """Relative paths are made absolute and the filename is preserved."""
    rel_path = "weights/test.pth"
    resolved = infer._resolve_checkpoint_path(rel_path)

    assert resolved.is_absolute()
    assert resolved.name == "test.pth"


def test_resolve_checkpoint_path_absolute(tmp_path, infer):
    """Absolute paths pass through unchanged."""
    abs_path = tmp_path / "absolute_test.pth"
    resolved = infer._resolve_checkpoint_path(abs_path)

    assert resolved == abs_path


def test_resolve_checkpoint_path_relative_anchor(infer):
    """Relative paths resolve relative to the project root, not the cwd."""
    project_root = Path(__file__).resolve().parents[1]
    resolved = infer._resolve_checkpoint_path("weights/test.pth")

    assert resolved == project_root / "weights" / "test.pth"


def test_build_transform_grayscale_shape(infer):
    """Grayscale transform produces a (1, H, W) tensor."""
    tf = infer.build_transform(image_size=16, mean=(0.5,), std=(0.5,), grayscale=True)
    img = Image.new("RGB", (32, 32))
    tensor = tf(img)

    assert tensor.shape == (1, 16, 16)


def test_build_transform_rgb_shape(infer):
    """RGB transform produces a (3, H, W) tensor."""
    tf = infer.build_transform(
        image_size=(32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        grayscale=False,
    )
    img = Image.new("RGB", (64, 64))
    tensor = tf(img)

    assert tensor.shape == (3, 32, 32)


def test_build_transform_int_and_tuple_size_equivalent(infer):
    """build_transform accepts both int and (H, W) tuple for square sizes."""
    tf_int = infer.build_transform(image_size=28, mean=(0.5,), std=(0.5,), grayscale=True)
    tf_tuple = infer.build_transform(image_size=(28, 28), mean=(0.5,), std=(0.5,), grayscale=True)
    img = Image.new("L", (56, 56))

    assert tf_int(img).shape == tf_tuple(img).shape


def test_inference_factory_known_alias(infer):
    """Factory returns an Inference instance for a known model alias."""
    with patch.object(infer, "load_model") as mock_load:
        mock_load.return_value = MagicMock(spec=infer.torch.nn.Module)
        inf = infer.InferenceFactory.create("mnist")

    assert isinstance(inf, infer.Inference)


def test_inference_factory_unknown_alias(infer):
    """Factory raises ValueError for an unrecognised model alias."""
    with pytest.raises(ValueError, match="Unknown model"):
        infer.InferenceFactory.create("invalid-model")


@patch("PIL.Image.open")
def test_inference_predict_correct_class(mock_open, infer):
    """predict() returns the class index with the highest logit."""
    mock_open.return_value = Image.new("L", (28, 28))

    mock_model = DummyModel()
    mock_logits = infer.torch.zeros((1, 10))
    # Correctly set index 7 to the highest value via tuple key.
    mock_logits[0, 7] = 5.0
    mock_model.return_value = mock_logits

    transform = lambda x: infer.torch.randn(1, 28, 28)
    inf = infer.Inference(model=mock_model, transform=transform, device="cpu")

    assert inf.predict("fake_path.png") == 7
    mock_model.assert_called_once()


@patch("PIL.Image.open")
def test_inference_predict_calls_model_once(mock_open, infer):
    """predict() invokes the model exactly once per image."""
    mock_open.return_value = Image.new("L", (28, 28))

    mock_model = DummyModel()
    mock_model.return_value = infer.torch.zeros((1, 10))

    inf = infer.Inference(
        model=mock_model,
        transform=lambda x: infer.torch.randn(1, 28, 28),
        device="cpu",
    )
    inf.predict("fake_path.png")

    mock_model.assert_called_once()


def test_run_inference_integration(tmp_path, infer):
    """run_inference returns a prediction for each image found in the directory."""
    _save_image(tmp_path / "test.png")

    with patch.object(infer.InferenceFactory, "create") as mock_factory:
        mock_inf = MagicMock()
        mock_inf.predict.return_value = 3
        mock_factory.return_value = mock_inf

        output = infer.run_inference(model="mnist", input_path=tmp_path)

    assert len(output) == 1
    assert output[0] == 3


def test_run_inference_empty_directory(tmp_path, infer):
    """run_inference raises ValueError for a directory with no valid images."""
    with patch.object(infer.InferenceFactory, "create") as mock_factory:
        mock_factory.return_value = MagicMock()
        with pytest.raises(ValueError, match="No valid image files found"):
            infer.run_inference(model="mnist", input_path=tmp_path)