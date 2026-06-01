# Adding a New Dataset

Use `MNISTDataModule` in `src/data.py` as the reference pattern.

## Steps

1. If dataset is avaialble in torchvision use the same format as MNIST other add the dataset name and source to `datasets/dataset_links.txt` and use accordingly. 

2. Add a data module in `src/data.py` that inherits from `BaseDataModule`.

   Add these methods:

   - `__init__(...)`
   - `train_loader()`
   - `test_loader()`

3. Add a loader helper in `src/data.py` to make loading the train and test dataloading easy during training.

   Template:

   ```python
   def get_<dataset_name>_loaders(data_dir="datasets", batch_size=64, num_workers=2):
       data_module = <DatasetName>DataModule(
           data_dir=data_dir,
           batch_size=batch_size,
           num_workers=num_workers,
       )
       return data_module.train_loader(), data_module.test_loader()
   ```

4. Update `__all__` in `src/data.py`.

   Add:

   - `<DatasetName>DataModule`
   - `get_<dataset_name>_loaders`

5. Check `src/models.py`.
   If the dataset has a different image size, number of channels, or number of
   classes, add a matching model class such as `<DatasetName>Net`.

6. Add training support in `src/training.py`.
   Use the new data module and the correct model.

   Typical names:

   - `<DatasetName>Trainer`
   - `train_<dataset_name>`

7. Add inference support in `src/inference.py`.

   Typical names:

   - `load_<dataset_name>_model`
   - `<DatasetName>Inference`
   - `predict_<dataset_name>`

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