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
├── datasets/
├── src/
  ├── data.py            # Dataset loaders (per-customer and general)
  ├── evaluation.py      # Evaluation script (precision, recall, speed)
  ├── inference.py       # Entry point for running predictions
  ├── instructions.md    # Instructions to use the scripts and extend the code
  ├── models.py          # Model definitions (per-customer and general)
  └── training.py        # General training script
├── tests/
  └── test_training.py   # Testing of the basic functionalities of src/training.py
├── weights/             # Trained model weights (trained on LUMI)
├── README.md            #
└── environment.yml      # Environment configuration file      
```

---

## Documentation

Full documentation is available at [https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/](https://fys-8805-collaborative-coding.github.io/Collaborative-Coding-Exam/) (GitHub Pages).

---

## Contributing

We use a branch → pull request → review workflow. All changes to `main` require at least one approved review — direct pushes are not allowed.

1. Open an issue describing your change
2. Create a branch and commit your work (reference the issue, e.g. `fixes #5`)
3. Open a pull request towards `main`
4. Get a review, address feedback, then merge

See [CONTRIBUTION.md](CONTRIBUTION.md) for the full guide.

---

## Authors

**Citation:**

---

## License

[MIT](LICENSE)
