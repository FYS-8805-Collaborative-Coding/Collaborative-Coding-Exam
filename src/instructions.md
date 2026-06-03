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
   Add an entry to the `PredictorFactory` registry in `src/inference.py`, mapping your dataset name to its model, image size, and normalization constants.
   NB: Make sure inference transforms match the training transforms.

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