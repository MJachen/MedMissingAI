from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    image: torch.Tensor,
    modality_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return class probabilities.

    Input image shape can be [M,D,H,W] or [B,M,D,H,W].
    Output shape is [B,num_classes].
    """

    model.eval()
    if image.ndim == 4:
        image = image.unsqueeze(0)
    if modality_mask.ndim == 1:
        modality_mask = modality_mask.unsqueeze(0)

    logits = model(image.to(device), modality_mask.to(device))
    return torch.softmax(logits, dim=1).cpu()

