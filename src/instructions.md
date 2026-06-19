# Adding a New Dataset

Use `MNISTDataModule` in `src/data.py` as the reference pattern.

## Steps

1. If the dataset is available in `torchvision`, create a new data module in `src/data.py` that inherits from `TorchvisionDataModule`. Otherwise, inherit from `BaseDataModule` and implement `train_loader()` and `test_loader()`.

2. Register the new data module in `src/data.py` by adding it to the `DATA_MODULES` dictionary. This allows it to be used automatically by the generic `get_loaders` function.

3. Update `__all__` in `src/data.py`.

   Add:
   - `<DatasetName>DataModule`

4. Check `src/models.py`.
   If the dataset has a different image size, number of channels, or number of
   classes, add a matching model class such as `<DatasetName>Net`.

5. Add training support in `src/training.py`.
   Add an entry to the `DATASET_REGISTRY` in `src/training.py`, referencing your new data module by its string name and the appropriate model class.

6. Add inference support in `src/inference.py`.
   Add an entry to `INFERENCE_REGISTRY` in `src/inference.py`, mapping your dataset name to its model, image size, and normalization constants.
   NB: Make sure inference transforms match the training transforms.

## Training Factory

Training uses a factory pattern. This means the command line dataset name is
looked up in `DATASET_REGISTRY`.

For each new dataset:

1. Import the data module and model class in `src/training.py`.

   Example:

   ```python
   from .data import MNISTDataModule, USPSDataModule
   from .models import MNISTNet, USPSNet
   ```

2. Add a dataset-specific trainer only if needed.

   If the dataset can use the normal training loop, use `Trainer`. If the
   dataset needs special training logic, add a class like this:

   ```python
   class USPSTrainer(Trainer):
       ...
   ```

3. Add the dataset to `DATASET_REGISTRY`.

   Example:

   ```python
   DATASET_REGISTRY = {
       "mnist": DatasetSpec("mnist", MNISTNet, "weights/mnist.pth", **DATASET_STATS["mnist"]),
       "usps":  DatasetSpec("usps",  USPSNet,  "weights/usps.pth",  **DATASET_STATS["usps"]),
   }
   ```

   The first argument is the data module string key (must match the key in `DATA_MODULES`). Stats are pulled from `DATASET_STATS` in `src/constants.py` — add your dataset there first.

4. Make sure the checkpoint path matches the dataset.

   Example:

   ```text
   weights/usps.pth
   ```

5. Train with the dataset name.

   Example:

   ```bash
   python -m src.training --dataset usps --epochs 1 --batch-size 64
   ```

   To save the checkpoint somewhere else:

   ```bash
   python -m src.training --dataset usps --epochs 5 --checkpoint-path path/to/usps.pth
   ```

After a dataset is added to `DATASET_REGISTRY`, it is automatically available
as a command-line choice.

## Inference Factory

Inference uses a factory pattern, like training. This means the command line
model name is looked up in `INFERENCE_REGISTRY`.

For each new dataset:

1. Import the model class in `src/inference.py`.

   Example:

   ```python
   from .models import MNISTNet, USPSNet
   ```

2. Add the dataset to `INFERENCE_REGISTRY`.

   Example:

   ```python
   INFERENCE_REGISTRY = {
       "mnist": InferenceSpec(
           MNISTNet, "weights/mnist.pth",
           image_size=DATASET_STATS["mnist"]["image_size"],
           mean=DATASET_STATS["mnist"]["mean"],
           std=DATASET_STATS["mnist"]["std"],
           grayscale=DATASET_STATS["mnist"]["grayscale"],
       ),
       "usps": InferenceSpec(
           USPSNet, "weights/usps.pth",
           image_size=DATASET_STATS["usps"]["image_size"],
           mean=DATASET_STATS["usps"]["mean"],
           std=DATASET_STATS["usps"]["std"],
           grayscale=DATASET_STATS["usps"]["grayscale"],
       ),
   }
   ```

   Use `DATASET_STATS` from `src/constants.py` to keep normalization consistent with training.

3. Make sure the checkpoint file exists.

   Example:

   ```text
   weights/usps.pth
   ```

5. Run inference with the dataset name.

   Example:

   ```bash
   python -m src.inference --model usps --input path/to/images
   ```

   To use a different checkpoint path:

   ```bash
   python -m src.inference --model usps --input path/to/images --checkpoint-path path/to/usps.pth
   ```

After a dataset is added to `INFERENCE_REGISTRY`, it is automatically available
as a command-line choice.

## Evaluation

Evaluation does not require a separate registry. It reuses `DATASET_REGISTRY` (from `src/training.py`) for data loading and `INFERENCE_REGISTRY` (from `src/inference.py`) for the model. Once both registries are updated, the dataset is automatically available in the evaluation CLI.

Run evaluation with:

```bash
python -m src.evaluation --dataset <dataset_name>
```

To use a specific checkpoint:

```bash
python -m src.evaluation --dataset <dataset_name> --checkpoint-path weights/<dataset_name>.pth
```

The command reports macro-averaged precision, macro-averaged recall, and average inference speed per sample.

## Naming

Use consistent names based on the dataset name:

- Data module class: `<DatasetName>DataModule`
- Loader helper: `get_<dataset_name>_loaders`
- Model class: `<DatasetName>Net`
- Trainer class: `<DatasetName>Trainer`
- Training helper: `train_<dataset_name>`
- Inference class: `<DatasetName>Inference`
- Model loader: `load_<dataset_name>_model`
- Prediction helper: `predict_<dataset_name>`
- Checkpoint path: `weights/<dataset_name>.pth`


