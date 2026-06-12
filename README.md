# ACME Digit Classification

ACME Digit Classification is a machine learning framework for handwritten digit recognition developed as part of the FYS-8805 Collaborative Coding Exam at UiT.

The repository provides a unified interface for training, evaluating, and deploying digit classification models for three customer datasets:

- **Customer A:** MNIST
- **Customer B:** SVHN
- **Customer C:** USPS

---
## Installation

Install directly from GitHub [NEED TO BE FIXED]:

```bash
pip install git+https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam.git
```

Alternatively, clone the repository via SSH:

```bash
git clone git@github.com:FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam.git
```

---
# Quick start: run interference

Suggested way to get inference from a trained model (example):

```python
from src import run_inference

results = run_inference(model="model-a", input_path="path/to/your/data")
print(results)
```

Or from the command line:

```bash
python -m src.inference --model model-a --input datasets/inference/mnist_test_0_label_7.png
```
Trained model weights are stored in the `weights/` directory and are loaded automatically during inference.

---
## Model Cards

### Model A — `model-a`

Brief description of what this model does and what problem it solves. 

| | |
|---|---|
| **Architecture** | ...|
| **Training data** | ... |
| **Intended use** | ... |
| **Limitations** | ... |

**Performance:**

| Metric | Value |
|---|---|
| Precision | 0.00 |
| Recall | 0.00 |
| Speed (inference) | 0.00 ms / sample |

---

### Model B — `model-b`

Brief description of what this model does and what problem it solves.

| | |
|---|---|
| **Architecture** | ... |
| **Training data** | ... |
| **Intended use** | ... |
| **Limitations** | ... |

**Performance:**

| Metric | Value |
|---|---|
| Precision | 0.00 |
| Recall | 0.00 |
| Speed (inference) | 0.00 ms / sample |

---
### Model C — `model-c`

Brief description of what this model does and what problem it solves.

| | |
|---|---|
| **Architecture** | ... |
| **Training data** | ... |
| **Intended use** | ... |
| **Limitations** | ... |

**Performance:**

| Metric | Value |
|---|---|
| Precision | 0.00 |
| Recall | 0.00 |
| Speed (inference) | 0.00 ms / sample |

---

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
