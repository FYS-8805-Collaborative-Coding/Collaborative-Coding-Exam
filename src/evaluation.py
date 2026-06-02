"""General evaluation for the ACME digit classifiers.
 
Provides a single :func:`evaluate` function that works for any customer's
model and dataloader, reporting classification quality (precision, recall)
and inference speed.
"""
 
import time
 
import torch
from sklearn.metrics import precision_score, recall_score
 
 
def evaluate(model, dataloader):
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
 
    with torch.no_grad():
        for images, labels in dataloader:
            start = time.perf_counter()
            outputs = model(images)
            total_time += time.perf_counter() - start
            all_preds.extend(outputs.argmax(dim=1).tolist())
            all_targets.extend(labels.tolist())
 
    precision = precision_score(all_targets, all_preds, average="micro")
    recall = recall_score(all_targets, all_preds, average="micro")
 
    speed_ms = 1000.0 * total_time / max(len(all_preds), 1)
 
    return {"precision": precision, "recall": recall, "speed_ms": speed_ms}