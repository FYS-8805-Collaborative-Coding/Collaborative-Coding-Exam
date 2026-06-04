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
   More details provided below. 


7. Add inference support in `src/inference.py`.
   More details provided below. 
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
       "mnist": DatasetSpec(MNISTDataModule, MNISTNet, "weights/mnist.pth", trainer_cls=MNISTTrainer),
       "usps": DatasetSpec(USPSDataModule, USPSNet, "weights/usps.pth"),
   }
   ```

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
   python -m src.training --dataset usps --checkpoint-path path/to/usps.pth
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

2. Add a dataset-specific inference class.

   Example:

   ```python
   class USPSInference(BaseInference):
       ...
   ```

   Put the correct image transforms inside this class. The transforms should
   match the dataset and model. For example, USPS uses 16x16 images, while
   MNIST uses 28x28 images.

3. Add the dataset to `INFERENCE_REGISTRY`.

   Example:

   ```python
   INFERENCE_REGISTRY = {
       "mnist": InferenceSpec(MNISTNet, "weights/mnist.pth", inference_cls=MNISTInference),
       "usps": InferenceSpec(USPSNet, "weights/usps.pth", inference_cls=USPSInference),
   }
   ```

4. Make sure the checkpoint file exists.

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
