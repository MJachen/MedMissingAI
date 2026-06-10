from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from medmissingai.data.dataset import MissingModalityDataset
from medmissingai.data.manifest import ManifestSpec, filter_split, load_manifest
from medmissingai.data.transforms import VolumePreprocessor, VolumeTransformConfig
from medmissingai.models.baseline import MissingModalityCNN
from medmissingai.training.engine import evaluate, save_checkpoint, train_one_epoch
from medmissingai.training.losses import build_loss
from medmissingai.utils.config import get_save_dir, load_experiment_config
from medmissingai.utils.device import resolve_device
from medmissingai.utils.io import write_json
from medmissingai.utils.run_metadata import save_run_metadata
from medmissingai.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the missing-modality baseline.")
    parser.add_argument("--config", default=None, help="Optional legacy combined config.")
    parser.add_argument("--data-config", default="configs/data_local.yaml")
    parser.add_argument("--train-config", default="configs/train_local.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_paths = load_experiment_config(
        config_path=args.config,
        data_config_path=args.data_config,
        train_config_path=args.train_config,
        model_config_path=args.model_config,
    )
    seed_everything(int(config.get("seed", 42)))
    save_dir = get_save_dir(config)
    save_run_metadata(save_dir, config_paths)

    modalities = tuple(config["data"]["modalities"])
    spec = ManifestSpec(modalities=modalities)
    frame = load_manifest(config["data"]["manifest"], spec)

    transform_config = VolumeTransformConfig(
        target_shape=tuple(config["data"]["target_shape"]),
        zscore=bool(config["data"].get("zscore", True)),
    )
    preprocessor = VolumePreprocessor(transform_config)

    train_set = MissingModalityDataset(filter_split(frame, "train", spec), spec, preprocessor)
    val_set = MissingModalityDataset(filter_split(frame, "val", spec), spec, preprocessor)

    device = resolve_device(config["training"].get("device", "auto"))
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = MissingModalityCNN(
        num_modalities=len(modalities),
        num_classes=int(config["model"]["num_classes"]),
        use_modality_mask=bool(config["model"].get("use_modality_mask", True)),
        base_channels=int(config["model"].get("base_channels", 16)),
    ).to(device)
    criterion = build_loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )

    best_metric = -1.0
    history: list[dict[str, float | int]] = []
    epochs = int(config["training"]["epochs"])

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=int(config["model"]["num_classes"]),
            desc="val",
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(row)

        score = float(val_metrics.get("accuracy", -val_metrics.get("loss", 0.0)))
        if score >= best_metric:
            best_metric = score
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, val_metrics, config)

    write_json(save_dir / "history.json", {"history": history})


if __name__ == "__main__":
    main()
