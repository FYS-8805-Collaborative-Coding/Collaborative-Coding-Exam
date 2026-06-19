"""General evaluation for the ACME digit classifiers.
 
Provides a single :func:`evaluate` function that works for any customer's
model and dataloader, reporting classification quality (precision, recall)
and inference speed.
"""
 
import warnings
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*", category=UserWarning)

import time
import argparse
from typing import Callable, Any, Iterable

import torch
from sklearn.metrics import precision_score, recall_score

from .data import DATA_MODULES
from .training import DATASET_REGISTRY
from .utils import setup_logging, get_logger
from .inference import InferenceFactory, BaseInference

logger = get_logger("evaluation")

DEFAULT_METRICS = {
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"),
}

def evaluate(
    inference: BaseInference, 
    dataloader: Iterable, 
    metrics: dict[str, Callable[[Any, Any], float]] | None = None
):
    """Evaluate a trained model on a dataset.

    Runs ``inference`` over every batch in ``dataloader`` and reports
    classification quality plus average forward-pass time per sample.

    Parameters
    ----------
    inference : BaseInference
        Inference instance whose underlying model handles batch prediction.
    dataloader : Iterable
        Yields ``(images, labels)`` batches.
    metrics : dict, optional
        Mapping of metric name to ``(y_true, y_pred) -> float``. Defaults to
        precision and recall (macro-averaged).

    Returns
    -------
    dict
        Keys ``"precision"``, ``"recall"`` (macro-averaged), and
        ``"speed_ms"`` (average forward-pass time per sample, in milliseconds).
    """
    metrics = metrics or DEFAULT_METRICS
    all_preds, all_targets = [], []
    total_time = 0.0

    for images, labels in dataloader:
        start = time.perf_counter()
        preds = inference.predict_batch_tensor(images)          # ← always safe
        total_time += time.perf_counter() - start
        all_preds.extend(preds)
        all_targets.extend(labels.tolist())

    results = {name: fn(all_targets, all_preds) for name, fn in metrics.items()}
    results["speed_ms"] = 1000.0 * total_time / max(len(all_preds), 1)
    return results

def build_arg_parser():
    """Construct the CLI argument parser for ``python -m src.evaluation``."""
    parser = argparse.ArgumentParser(description="Evaluate a trained dataset model on its test set.")
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Dataset to evaluate on.",
    )
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint to load (defaults to the dataset's registered path).")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--data-dir", default="datasets", help="Directory containing the datasets.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--device", default="cpu", help="Device to use, e.g. cpu, cuda, or mps.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser

def main(argv=None):
    """CLI entry point: parse arguments, build the inference and test loader, and log the metrics."""
    args = build_arg_parser().parse_args(argv)

    setup_logging(args.log_level)

    device = args.device

    spec = DATASET_REGISTRY[args.dataset]
    checkpoint = args.checkpoint_path or spec.default_checkpoint

    logger.info("Start  dataset=%s device=%s checkpoint=%s", args.dataset, device, checkpoint)

    # Create the inference instance to ensure production-spec settings (mean, std, transform) are used
    inference = InferenceFactory.create(
        args.dataset,
        checkpoint_path=checkpoint,
        device=device
    )

    data_module = DATA_MODULES[spec.data_module_name](
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = data_module.test_loader()

    result = evaluate(inference, test_loader)
    
    metrics_log = "  ".join([f"{k}={v:.4f}" if k != "speed_ms" else f"{k}={v:.3f} ms/sample" 
                             for k, v in result.items()])
    logger.info(
        "Results  %s", metrics_log
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())