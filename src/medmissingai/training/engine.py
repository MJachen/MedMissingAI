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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        modality_mask = batch["modality_mask"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image, modality_mask)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * label.size(0)
        total_count += label.size(0)

    return {"loss": total_loss / max(total_count, 1)}


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

