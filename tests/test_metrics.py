from __future__ import annotations

import numpy as np

from medmissingai.training.metrics import binary_classification_metrics


def test_binary_classification_metrics_confusion_terms() -> None:
    labels = np.array([0, 0, 1, 1])
    positive_probs = np.array([0.2, 0.8, 0.7, 0.4])

    metrics = binary_classification_metrics(labels, positive_probs, threshold=0.5)

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
