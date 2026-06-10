from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            if num_classes == 2:
                auc = float(roc_auc_score(labels, probs[:, 1]))
                if np.isfinite(auc):
                    metrics["roc_auc"] = auc
            else:
                auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
                if np.isfinite(auc):
                    metrics["roc_auc_ovr"] = auc
    except ValueError:
        pass

    return metrics


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
