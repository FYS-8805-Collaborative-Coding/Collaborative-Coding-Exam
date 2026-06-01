# Collaborative-Coding-Exam

Short, clear description of what this repo does and who it's for. Two or three sentences max.

---

## Installation

via SSH

```bash
git clone git@github.com:FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam.git
```

(or via pip?)

---

## Running Inference

Suggested way to get inference from a trained model (example):

```python
from inference import run_inference

results = run_inference(model="model-a", input="path/to/your/data")
print(results)
```

Or from the command line:

```bash
python inference.py --model model-a --input path/to/your/data
```

Trained model weights are stored in the `weights/` folder and are loaded automatically.

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

## Repository Structure

```
├── models.py          # Model definitions (per-customer and general)
├── data.py            # Dataset loaders (per-customer and general)
├── training.py        # General training script
├── evaluation.py      # Evaluation script (precision, recall, speed)
├── inference.py       # Entry point for running predictions
├── weights/           # Trained model weights (trained on LUMI)
└── docs/              # Sphinx documentation
```

---

## Documentation

Full documentation is available at [https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/](https://https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/) (GitHub Pages).

---

## Contributing

We use a branch → pull request → review workflow. All changes to `main` require at least one approved review — direct pushes are not allowed.

1. Open an issue describing your change
2. Create a branch and commit your work (reference the issue, e.g. `fixes #5`)
3. Open a pull request towards `main`
4. Get a review, address feedback, then merge

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## Authors

**Citation:**

[*citation placeholder*]

See also [CITATION.cff](CITATION.cff).

---

## License

[MIT](LICENSE)