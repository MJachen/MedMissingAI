from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from medmissingai.training.metrics import classification_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality_dropout_prob: float = 0.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        modality_mask = batch["modality_mask"].to(device)
        label = batch["label"].to(device)
        if modality_dropout_prob > 0:
            image, modality_mask = apply_modality_dropout(
                image,
                modality_mask,
                modality_dropout_prob,
            )

        optimizer.zero_grad(set_to_none=True)
        logits = model(image, modality_mask)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * label.size(0)
        total_count += label.size(0)

    return {"loss": total_loss / max(total_count, 1)}


def apply_modality_dropout(
    image: torch.Tensor,
    modality_mask: torch.Tensor,
    dropout_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dropout_prob <= 0:
        return image, modality_mask
    if dropout_prob >= 1:
        raise ValueError("modality_dropout_prob must be less than 1.0")

    present = modality_mask > 0
    drop = torch.rand_like(modality_mask) < dropout_prob
    keep = modality_mask.clone()
    keep[present & drop] = 0.0

    empty_rows = keep.sum(dim=1) == 0
    if empty_rows.any():
        for row_idx in torch.nonzero(empty_rows, as_tuple=False).flatten():
            present_indices = torch.nonzero(present[row_idx], as_tuple=False).flatten()
            if len(present_indices) > 0:
                chosen = present_indices[
                    torch.randint(len(present_indices), (1,), device=present_indices.device)
                ]
                keep[row_idx, chosen] = 1.0

    image = image * keep.view(keep.shape[0], keep.shape[1], 1, 1, 1)
    return image, keep


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    desc: str = "eval",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        image = batch["image"].to(device)
        modality_mask = batch["modality_mask"].to(device)
        label = batch["label"].to(device)

        logits = model(image, modality_mask)
        loss = criterion(logits, label)

        total_loss += float(loss.item()) * label.size(0)
        total_count += label.size(0)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())

    if not all_logits:
        return {"loss": 0.0}

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = classification_metrics(logits_np, labels_np, num_classes)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )
