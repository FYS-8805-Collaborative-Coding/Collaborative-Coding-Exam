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
