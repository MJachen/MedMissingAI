from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.exceptions import UndefinedMetricWarning


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    probs = softmax(logits)
    if num_classes == 2:
        return binary_classification_metrics(labels, probs[:, 1])

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


def binary_classification_metrics(
    labels: np.ndarray,
    positive_probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    positive_probs = np.asarray(positive_probs).astype(float)
    preds = (positive_probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]

    specificity = _safe_divide(tn, tn + fp)
    sensitivity = _safe_divide(tp, tp + fn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        balanced_accuracy = float(balanced_accuracy_score(labels, preds))
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "label_counts": _counts(labels),
        "prediction_counts": _counts(preds),
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            auc = float(roc_auc_score(labels, positive_probs))
            if np.isfinite(auc):
                metrics["roc_auc"] = auc
    except ValueError:
        pass

    return metrics


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
