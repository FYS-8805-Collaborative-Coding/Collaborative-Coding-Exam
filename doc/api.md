# API Reference

This page documents the public Python API exposed by the `ccexam` package.

---

## Inference (`src.inference`)

### `run_inference`

```python
from src.inference import run_inference

run_inference(
    model: str,
    input_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> int | list[int]
```

Run inference on one image or a folder of images.

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Dataset alias: `"mnist"`, `"usps"`, `"svhn"` |
| `input_path` | `str \| Path` | Path to a single image file or a directory of images |
| `checkpoint_path` | `str \| Path \| None` | Override the default checkpoint file (optional) |
| `device` | `str \| None` | Torch device, e.g. `"cpu"`, `"cuda"`, `"mps"`. Auto-detected if `None` |

**Returns** a single `int` label when `input_path` is a file, or a `list[int]` (sorted by filename) when it is a directory.

**Example**

```python
from src.inference import run_inference

label = run_inference(model="usps", input_path="digit.png")
print(label)  # e.g. 7

labels = run_inference(model="mnist", input_path="folder/")
print(labels)  # e.g. [3, 1, 4, 1, 5]
```

---

### `InferenceFactory`

```python
from src.inference import InferenceFactory

predictor = InferenceFactory.create(model_name: str, **kwargs)
predictor.predict(image)  # -> int
```

Creates a configured `Inference` instance for the given model name. Accepts the same keyword arguments as `run_inference` (`device`, `checkpoint_path`, etc.).

---

### `INFERENCE_REGISTRY`

Dictionary mapping model/dataset names to `InferenceSpec` objects that define the model class, default checkpoint path, image size, and normalization constants.

| Key | Model class | Image size | Channels |
|---|---|---|---|
| `"mnist"` / `"model-a"` | `MNISTNet` | 28×28 | Grayscale |
| `"usps"` / `"model-b"` | `USPSNet` | 16×16 | Grayscale |
| `"svhn"` / `"model-c"` | `SVHNNet` | 32×32 | RGB |

---

### `write_results`

```python
from src.inference import write_results

write_results(results: dict[Path, int], output_path: str | Path) -> Path
```

Write a `{image_path: label}` dictionary to a `.csv` or `.txt` file. The output file is never overwritten — a numbered copy is created instead (`predictions_1.csv`, …).

---

## Models (`src.models`)

All model classes are `torch.nn.Module` subclasses callable with an image batch tensor.

### `MNISTNet`

CNN for 28×28 grayscale MNIST digits. Two convolutional blocks followed by a two-layer classifier head (128 hidden units, 10 outputs).

```python
from src.models import MNISTNet

model = MNISTNet()
logits = model(batch)  # batch shape: (N, 1, 28, 28)
```

### `USPSNet`

Same architecture as `MNISTNet`, adapted for 16×16 grayscale USPS digits.

```python
from src.models import USPSNet

model = USPSNet()
logits = model(batch)  # batch shape: (N, 1, 16, 16)
```

### `SVHNNet`

Deeper CNN for 32×32 RGB SVHN digits. Three convolutional blocks with batch normalization, followed by a dropout-regularized classifier head.

```python
from src.models import SVHNNet

model = SVHNNet(num_classes=10, dropout=0.3)
logits = model(batch)  # batch shape: (N, 3, 32, 32)
```

| Parameter | Default | Description |
|---|---|---|
| `num_classes` | `10` | Number of output classes |
| `dropout` | `0.3` | Dropout probability in the classifier head |

---

## Training (`src.training`)

### `train`

```python
from src.training import train

metrics = train(dataset: str = "mnist", **kwargs) -> dict
```

Convenience entry point. Selects the correct data module and model from `DATASET_REGISTRY` and runs training.

| Keyword argument | Default | Description |
|---|---|---|
| `epochs` | `1` | Number of training epochs |
| `lr` | `1e-3` | Learning rate (Adam optimizer) |
| `batch_size` | `64` | Mini-batch size |
| `checkpoint_path` | registry default | Where to save the `.pth` checkpoint |
| `data_dir` | `"datasets"` | Root directory for dataset downloads |
| `device` | auto | `"cpu"`, `"cuda"`, or `"mps"` |
| `loss_fn` | `CrossEntropyLoss` | Custom `torch.nn` loss instance |

**Returns** a dict with keys `loss`, `accuracy`, `val_loss`, `val_accuracy`, `epochs`, `checkpoint_path`.

**Example**

```python
from src.training import train

metrics = train(dataset="usps", epochs=5, batch_size=128)
print(metrics["accuracy"])
```

---

### `DATASET_REGISTRY`

Maps dataset names to `DatasetSpec` objects (data module class, model class, default checkpoint path).

| Key | Model | Default checkpoint |
|---|---|---|
| `"mnist"` | `MNISTNet` | `weights/mnist.pth` |
| `"usps"` | `USPSNet` | `weights/usps.pth` |
| `"svhn"` | `SVHNNet` | `weights/svhn.pth` |

---

## Data (`src.data`)

### `get_loaders`

```python
from src.data import get_loaders

train_loader, test_loader = get_loaders(
    dataset: str = "mnist",
    batch_size: int = 64,
    data_dir: str = "datasets",
    num_workers: int = 2,
    download: bool = True,
)
```

Returns `(train_loader, test_loader)` as `torch.utils.data.DataLoader` objects for a registered dataset. Dataset files are downloaded automatically on first use.

**Example**

```python
from src.data import get_loaders

train_loader, test_loader = get_loaders(dataset="svhn", batch_size=128)
for images, labels in train_loader:
    ...
```

---

### Data modules

Each dataset has a dedicated data module that can also be used directly:

| Class | Dataset | Default normalization |
|---|---|---|
| `MNISTDataModule` | MNIST | mean=0.1307, std=0.3081 |
| `USPSDataModule` | USPS | mean=0.2471, std=0.2994 |
| `SVHNDataModule` | SVHN | mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970) |

All modules expose `.train_loader()`, `.val_loader()`, and `.test_loader()` methods.
