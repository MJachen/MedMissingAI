from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def build_loss(
    class_weights: str | Sequence[float] | None = None,
    labels: Sequence[int] | None = None,
    num_classes: int | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    weight = None
    if class_weights is not None:
        if isinstance(class_weights, str):
            if class_weights != "balanced":
                raise ValueError("class_weights must be `balanced`, a numeric list, or null")
            if labels is None or num_classes is None:
                raise ValueError("balanced class_weights requires labels and num_classes")
            weight = _balanced_class_weights(labels, num_classes)
        else:
            weight = torch.tensor([float(value) for value in class_weights], dtype=torch.float32)
            if num_classes is not None and weight.numel() != num_classes:
                raise ValueError(
                    f"class_weights length must match num_classes={num_classes}, got {weight.numel()}"
                )

    if weight is not None and device is not None:
        weight = weight.to(device)
    return nn.CrossEntropyLoss(weight=weight)


def _balanced_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        label_idx = int(label)
        if label_idx < 0 or label_idx >= num_classes:
            raise ValueError(f"label {label_idx} is outside [0, {num_classes})")
        counts[label_idx] += 1.0

    total = counts.sum()
    weights = torch.ones(num_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = total / (float(num_classes) * counts[present])
    return weights
