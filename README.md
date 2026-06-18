<p align="center">
    <img src="doc/_static/uit.png" alt="UiT The Arctic University of Norway" width="130">
</p>

<div align="center">
<h1 align="center">ACME Digit Classification</h1>
<h4 align="center"><em>A unified framework for handwritten-digit recognition — MNIST · SVHN · USPS</em></h4>
<h4 align="center"><em>FYS-8805 Collaborative Coding Exam · UiT The Arctic University of Norway</em></h4>
<h5 align="center"><em>Siyan Chen · Andrea Løkke · Riccardo Gelato · Rahul Baburajan · Keyne Oei · Myrthe Catteau</em></h5>
</div>

<p align="center">
  <img src="doc/_static/collab.png" alt="Collaborating on digit classification under the northern lights in Tromsø" width="820">
</p>

<p align="center">
    <a href="https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam/actions/workflows/test.yml"><img src="https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam/actions/workflows/test.yml/badge.svg" alt="CI"></a>
    <a href="https://pypi.org/project/ccexam/"><img src="https://img.shields.io/pypi/v/ccexam" alt="PyPI"></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
    <a href="https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam/blob/main/LICENSE"><img src="https://img.shields.io/github/license/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam" alt="License"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#quick-start-run-inference">Quick start</a> |
  <a href="#model-cards">Model Cards</a> |
  <a href="#documentation">Documentation</a> |
  <a href="#model-development">Development</a>
</p>

---
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
ccexam-infer --model svhn --input src/samples/svhn_digit_5.png

# Classify every image in a directory
ccexam-infer --model svhn --input src/samples

# Try it instantly on a bundled sample (no file needed) — see "Quick test" below
ccexam-infer --model svhn --input samples:svhn_digit_5.png

# Force CPU (e.g. on a laptop with no GPU)
ccexam-infer --model svhn --input mydigit.png --device cpu
```

Available models: `mnist`, `usps`, `svhn`. The trained model weights are bundled with the package and loaded automatically — no extra setup needed.

### Supported inputs

`--input` accepts a single file or a directory, and is flexible about formats:

- **Any image, detected by content — not by extension.** A real image is accepted even if it has the "wrong" extension or **no extension at all** (e.g. `--input mydigit`). A non-image is rejected with a error `Invalid input: …` message and a non-zero exit code.
- **ASCII-art digits in `.txt` files.** A digit drawn with characters (foreground vs. spaces) is auto-detected and converted to an image before inference — no flag needed:
  ```bash
  ccexam-infer --model mnist --input datasets/inference/ascii_digit_5.txt
  ```
- **Directories** skip any unreadable files (with a logged note) instead of aborting the whole run.

### Quick test with bundled samples

To try the package without supplying your own files, use the `samples:` scheme. It resolves to an example image shipped inside the package, so it works straight after `pip install`:

```bash
ccexam-infer --model svhn  --input samples:svhn_digit_5.png
ccexam-infer --model mnist --input samples:mnist_test_0_label_7.png
ccexam-infer --model mnist --input samples:ascii_digit_5.txt     # bundled ASCII sample
```

### Saving predictions to a file

By default, predictions are only printed to the terminal. Pass `--output` (`-o`)
to also write them to a file:

```bash
# Write predictions to results/predictions.csv (the results/ folder is created automatically)
ccexam-infer --model svhn --input src/samples --output results/predictions.csv
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
| Recall | 0.9904 |
| Inference time | 0.057 ms/sample (CPU), 0.051 ms/sample (MPS) |

---

### Model B — SVHN

CNN classifier for Street View House Numbers. It predicts a single cropped house-number digit (0–9) from a 32×32 RGB image.

| | |
|---|---|
| **Architecture** | 3-block CNN (per block: Conv-BN-ReLU ×2 + MaxPool, channels 3→32→64→128) followed by a 2-layer fully-connected head with dropout 0.3 |
| **Training data** | SVHN `train_32x32.mat` (~73k 32×32 RGB cropped-digit images), trained 5 epochs, Adam, lr 1e-3, batch size 64 |
| **Intended use** | Classifying house-number digits |
| **Limitations** | Single digits only (not multi-digit house numbers). Fixed 32×32 RGB input. Trained only 5 epochs with no data augmentation |

**Performance:**

| Metric | Value |
|---|---|
| Precision | 0.9436 |
| Recall | 0.9473 |
| Inference time | 0.292 ms/sample (CPU), 0.046 ms/sample (MPS) |

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
| Precision | 0.9709 |
| Recall | 0.9697|
| Inference time | 0.024 ms/sample (CPU), 0.058 ms/sample (MPS) |


## Documentation

More detailed documentation and examples are available at [https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/](https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/). Including instructions for:
- [testing](https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/development.html#training)
- [training](https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/development.html#training)
- [evaluation](https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/api.html#evaluation-src-evaluation)
- [usage of HPC (Lumi)](https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/lumi.html)
- [contributing to the code](https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/development.html#contributing-to-the-code)

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
