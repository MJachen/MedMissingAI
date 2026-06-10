from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def gradient_saliency(
    model: nn.Module,
    image: torch.Tensor,
    modality_mask: torch.Tensor,
    target_class: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Compute a simple gradient saliency heatmap.

    Input image shape: [M,D,H,W].
    Output heatmap shape: [D,H,W], normalized to [0,1].
    """

    model.eval()
    x = image.unsqueeze(0).to(device).requires_grad_(True)
    mask = modality_mask.unsqueeze(0).to(device)
    logits = model(x, mask)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    score = logits[:, target_class].sum()
    model.zero_grad(set_to_none=True)
    score.backward()

    saliency = x.grad.detach().abs().amax(dim=1).squeeze(0)
    saliency = saliency - saliency.min()
    saliency = saliency / saliency.max().clamp_min(1e-6)
    return saliency.cpu()


def save_middle_slice_overlay(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    output_path: str | Path,
    modality_index: int = 0,
    alpha: float = 0.45,
) -> None:
    """Save a PNG overlay for the middle depth slice."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth = image.shape[1]
    z = depth // 2
    base = image[modality_index, z].detach().cpu().numpy()
    heat = heatmap[z].detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(base, cmap="gray")
    plt.imshow(heat, cmap="hot", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()

