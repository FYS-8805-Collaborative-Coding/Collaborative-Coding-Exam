# Model development
## Training

You can run training from the repository root with the CLI. All arguments are entirely optional; running the command without any flags will automatically train on the default `mnist` dataset:

```bash
# Run with absolute defaults (MNIST, 1 epoch, batch size 64, automatic device selection)
python -m src.training

# Run with custom configuration overrides
python -m src.training --dataset mnist --epochs 5 --batch-size 32 --device cuda
```

The checkpoint is written to `src/weights/<dataset>.pth` by default. Training supports `mnist`, `usps`, and `svhn` out of the box; additional datasets can be registered in `src/training.py`.

### Using a config file

Instead of passing all flags on the command line, you can use a config file with `--config` (`-c`). The `configs/` directory contains ready-made configs for each dataset:

```bash
python -m src.training --config configs/mnist.cfg
python -m src.training --config configs/svhn.cfg
python -m src.training --config configs/usps.cfg
```

A config file is a plain `key=value` text file (lines starting with `#` are comments):

```
# configs/mnist.cfg
dataset=mnist
epochs=50
batch_size=32
```

Any CLI flag you pass explicitly will override the corresponding value in the config file — the priority order is **CLI flag > config file > built-in default**:

```bash
# Use the config but run only 5 epochs instead of 50
python -m src.training --config configs/mnist.cfg --epochs 5
```

## Evaluation

After training, evaluate a model on its test set from the repository root:

```bash
# Evaluate the default MNIST model on CPU
python -m src.evaluation --dataset mnist

# Evaluate with a custom checkpoint on NVIDIA GPU
python -m src.evaluation --dataset svhn --checkpoint-path weights/svhn.pth --device cuda

# Evaluate on Apple Silicon
python -m src.evaluation --dataset svhn --checkpoint-path weights/svhn.pth --device mps
```

The command prints macro-averaged precision, macro-averaged recall, and average
inference time per sample. The test data is downloaded automatically on the first
run to the directory given by `--data-dir` (default: `datasets/`).

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `mnist` | Dataset to evaluate on: `mnist`, `usps`, or `svhn`. |
| `--checkpoint-path` | *(registered default)* | Path to a `.pth` checkpoint file. |
| `--batch-size` | `256` | Number of samples per batch. |
| `--data-dir` | `datasets` | Root directory for dataset downloads. |
| `--num-workers` | `0` | Number of DataLoader worker processes. |
| `--device` | `cpu` | PyTorch device: `cpu`, `cuda` (NVIDIA GPU), or `mps` (Apple Silicon). |
| `--log-level` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`. |

---

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

**If you have write access to the repository:**

1. Open an issue describing your change
2. Create a branch and commit your work (reference the issue, e.g. `fixes #5`)
3. Open a pull request towards `main`
4. Get a review, address feedback, then merge

**If you do not have write access (external contributors):**

1. Fork the repository on GitHub
2. Clone your fork locally: `git clone https://github.com/<your-username>/Collaborative-Coding-Exam.git`
3. Create a branch in your fork and commit your work
4. Open a pull request from your fork towards `main` of this repository
5. Get a review, address feedback, then merge

See [CONTRIBUTION.md](https://github.com/FYS-8805-Collaborative-Coding/Collaborative-Coding-Exam/blob/main/CONTRIBUTION.md#contributing) for the full guide.

---
