"""General evaluation for the ACME digit classifiers.
 
Provides a single :func:`evaluate` function that works for any customer's
model and dataloader, reporting classification quality (precision, recall)
and inference speed.
"""
 
import time
import argparse

import torch
from sklearn.metrics import precision_score, recall_score

from src.data import DATA_MODULES
from src.training import DATASET_REGISTRY
from src.utils import setup_logging, get_logger

logger = get_logger("evaluation")


def evaluate(model, dataloader, device):
    """Evaluate a trained model on a dataset.

    Runs the model over every batch in ``dataloader``, collects the predicted
    and true labels, and reports classification quality together with the
    average inference time per sample.

    Parameters
    ----------
    model : torch.nn.Module
        A trained model. It is set to evaluation mode and called as
        ``model(images)``, returning class logits of shape
        ``(batch_size, num_classes)``.
    dataloader : Iterable
        Yields ``(images, labels)`` batches, where ``images`` is a float
        tensor and ``labels`` is an integer tensor of true classes.
    device : str or torch.device, optional
        Device to run evaluation on (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
        The model and each batch are moved here. Defaults to ``"cpu"``.
 
    Returns
    -------
    dict
        A dictionary with the keys:
 
        ``"precision"`` : float
            Precision, averaged across classes.
        ``"recall"`` : float
            Recall, averaged across classes.
        ``"speed_ms"`` : float
            Average inference time in milliseconds per sample.
 
    Notes
    -----
    Evaluation runs under :func:`torch.no_grad`, so no gradients are tracked.
    Inference time covers only the forward pass, not data loading.
    """
    model.eval()
    all_preds, all_targets = [], []
    total_time = 0.0

    # Move the model and each batch to the requested device.
    device = torch.device(device)
    model.to(device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            start = time.perf_counter()
            outputs = model(images)
            total_time += time.perf_counter() - start
            all_preds.extend(outputs.argmax(dim=1).tolist())
            all_targets.extend(labels.tolist())
 
    precision = precision_score(all_targets, all_preds, average="micro")
    recall = recall_score(all_targets, all_preds, average="micro")
 
    speed_ms = 1000.0 * total_time / max(len(all_preds), 1)
 
    return {"precision": precision, "recall": recall, "speed_ms": speed_ms}

def build_arg_parser():
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
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--device", default="cpu", help="Device to use, e.g. cpu, cuda, or mps.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser

def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    setup_logging(args.log_level)

    device = torch.device(args.device)

    spec = DATASET_REGISTRY[args.dataset]
    checkpoint = args.checkpoint_path or spec.default_checkpoint

    logger.info("Start  dataset=%s device=%s checkpoint=%s", args.dataset, device, checkpoint)

    model = spec.model_cls()
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    data_module = DATA_MODULES[spec.data_module_name](
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = data_module.test_loader()

    result = evaluate(model, test_loader, device=device)
    logger.info(
        "Results  precision=%.4f  recall=%.4f  speed=%.3f ms/sample",
        result["precision"], result["recall"], result["speed_ms"],
    )
    return 0

if __name__ == "__main__":
    main()