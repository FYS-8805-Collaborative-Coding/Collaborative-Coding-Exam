# ACME Digit Classification

[![CI](https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam/actions/workflows/test.yml/badge.svg)](https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/ccexam)](https://pypi.org/project/ccexam/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/github/license/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam)

ACME Digit Classification is a machine learning framework for handwritten digit recognition developed as part of the FYS-8805 Collaborative Coding Exam at UiT.

The repository provides a unified interface for training, evaluating, and deploying digit classification models for three customer datasets:

- **Customer A:** MNIST
- **Customer B:** SVHN
- **Customer C:** USPS

---
## Installation

Install the latest release from [PyPI](https://pypi.org/project/ccexam/):

```bash
pip install ccexam
```

To upgrade an existing installation to the latest release:

```bash
pip install --upgrade ccexam
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam.git
```

### For development

Clone the repository and create the conda environment from `environment.yml`:

```bash
git clone https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam.git
cd Collaborative-Coding-Exam
conda env create -f environment.yml
conda activate collaborative-coding-exam
```

---
# Quick start: run inference

After installing the package (`pip install ccexam`), run inference from the command line with the `ccexam-infer` command:

```bash
# Classify a single image
ccexam-infer --model svhn --input datasets/inference/svhn_digit_5.png

# Classify every image in a directory
ccexam-infer --model svhn --input datasets/inference

# Force CPU (e.g. on a laptop with no GPU)
ccexam-infer --model svhn --input mydigit.png --device cpu
```

Available models: `mnist`, `usps`, `svhn`. The trained model weights are bundled with the package and loaded automatically — no extra setup needed.

### Saving predictions to a file

By default, predictions are only printed to the terminal. Pass `--output` (`-o`)
to also write them to a file:

```bash
# Write predictions to results/predictions.csv (the results/ folder is created automatically)
ccexam-infer --model svhn --input datasets/inference --output results/predictions.csv
```

Notes:

- The path you give is used **exactly as written** — any folders in it (e.g. `results/`) are created for you.
- **Existing files are never overwritten.** If `results/predictions.csv` already exists, the next run writes `results/predictions_1.csv`, then `_2`, and so on, so previous results are preserved.
- The format follows the file extension: `.csv` writes an `image,prediction` table with a header row; `.txt` writes one `<image>\t<prediction>` line per image.

### Python API

```python
from ccexam import run_inference

# A single image returns one label
label = run_inference(model="svhn", input_path="digit.png")
print(label)     # 5

# A folder of images returns a list of labels (sorted by filename)
labels = run_inference(model="svhn", input_path="folder_of_digits/")
print(labels)    # [7, 2, 1, 0]
```

> When working from a repository checkout without installing, you can run the same thing as a module: `python -m src.inference --model svhn --input <path> --output results/predictions.csv`.

---
## Model Cards

### Model A — `MNIST`

Model A is an MNIST digit classifier for recognizing handwritten digits from
28x28 grayscale images.

| | |
|---|---|
| **Architecture** | `MNISTNet`: small CNN with two convolution/ReLU/max-pool blocks and a fully connected classifier with 128 unit hidden dimension. |
| **Training data** | MNIST handwritten digit training set. |
| **Intended use** | Classify MNIST-like digit images into classes 0-9. |
| **Limitations** | Intended for clean MNIST style grayscale digits; performance may drop on other image styles, noise, or non digit inputs. |

**Performance:**

| Metric | Value |
|---|---|
| Precision | 0.9905 |
| Recall | 0.9905 |
| Speed (inference) | 0.060 ms / sample |

---

### Model B — SVHN

CNN classifier for Street View House Numbers. It predicts a single cropped house-number digit (0–9) from a 32×32 RGB image.

| | |
|---|---|
| **Architecture** | 3-block CNN (per block: Conv-BN-ReLU ×2 + MaxPool, channels 3→32→64→128) followed by a 2-layer fully-connected head with dropout 0.3 |
| **Training data** | SVHN `train_32x32.mat` (~73k 32×32 RGB cropped-digit images), trained 5 epochs, Adam, lr 1e-3, batch size 64 |
| **Intended use** | Classifying house-number digits |
| **Limitations** | Single digits only (not multi-digit house numbers). Fixed 32×32 RGB input. Trained only 5 epochs with no data augmentation |

**Performance:** *(measured on the SVHN test set, `weights/svhn.pth`, LUMI-G / MI250x)*

| Metric | Value |
|---|---|
| Precision | 0.9465 |
| Recall | 0.9465 |
| Speed (inference) | 0.470 ms / sample |

---
### Model C — USPS

CNN classifier trained on the USPS (United States Postal Service) dataset consisting of ~10,000 images of handwritten digits.

| | |
|---|---|
| **Architecture** | `USPSNet`: small CNN with two convolution/ReLU/max-pool blocks and a fully connected classifier with 128 unit hidden dimension. |
| **Training data** | USPS (United States Postal Service) dataset consisting of ~10,000 16x16 grayscale images |
| **Intended use** | Classifying grayscale handwritten digits |
| **Limitations** | Limited amount of data, with fixed 16x16 grid. Only grayscale. |

**Performance:**

| Metric | Value |
|---|---|
| Precision | 0.9631 |
| Recall | 0.9631|
| Speed (inference) | 0.033 ms / sample |


## Documentation

More detailed documentation is available at [https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/](https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/).


---
# Model development
## Training

You can run training from the repository root with the CLI. All arguments are entirely optional; running the command without any flags will automatically train on the default `mnist` dataset:

```bash
# Run with absolute defaults (MNIST, 1 epoch, batch size 64, automatic device selection)
python -m src.training

# Run with custom configuration overrides
python -m src.training --dataset mnist --epochs 5 --batch-size 32 --device cuda
```

The checkpoint is written to `weights/mnist.pth` by default. The current
training entry point supports `mnist` and can be extended with more datasets
through the registry in `src/training.py`.

## Testing

Run the basic tests with:

```bash
pytest -q
```

The tests are lightweight and only validate the training CLI, argument parsing,
and factory wiring.

---

## Contributing to the code

We use a branch → pull request → review workflow. All changes to `main` require at least one approved review — direct pushes are not allowed.

1. Open an issue describing your change
2. Create a branch and commit your work (reference the issue, e.g. `fixes #5`)
3. Open a pull request towards `main`
4. Get a review, address feedback, then merge

See [CONTRIBUTION.md](CONTRIBUTION.md) for the full guide.

---

# Further use of the software
If you use this software in your research, teaching, or projects, please cite this repository. This project is released under the MIT License. You are free to use, modify, and distribute the software in accordance with the terms of the license.
## Citation

**APA:**

Chen, S., Løkke, A., Gelato, R., Baburajan, R., Oei, K., & Catteau, M. (2026). Collaborative Coding Exam (Version 1.1.0) [Computer software]. https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam

**BibTex:**

```
@software{Chen_Collaborative_Coding_Exam_2026,
author = {Chen, Siyan and Løkke, Andrea and Gelato, Riccardo and Baburajan, Rahul and Oei, Keyne and Catteau, Myrthe},
month = jun,
title = {{Collaborative Coding Exam}},
url = {https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam},
version = {1.1.0},
year = {2026}
}
```

## License

[MIT](LICENSE)
