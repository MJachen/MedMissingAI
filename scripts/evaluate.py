from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from medmissingai.data.dataset import MissingModalityDataset
from medmissingai.data.manifest import ManifestSpec, filter_split, load_manifest
from medmissingai.data.transforms import VolumePreprocessor, VolumeTransformConfig
from medmissingai.models.baseline import MissingModalityCNN
from medmissingai.training.engine import evaluate
from medmissingai.training.losses import build_loss
from medmissingai.utils.config import get_save_dir, load_experiment_config
from medmissingai.utils.device import resolve_device
from medmissingai.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", default=None, help="Optional legacy combined config.")
    parser.add_argument("--data-config", default="configs/data_local.yaml")
    parser.add_argument("--train-config", default="configs/train_local.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, _ = load_experiment_config(
        config_path=args.config,
        data_config_path=args.data_config,
        train_config_path=args.train_config,
        model_config_path=args.model_config,
    )
    save_dir = get_save_dir(config)
    checkpoint_path = Path(args.checkpoint or save_dir / "best.pt")

    modalities = tuple(config["data"]["modalities"])
    spec = ManifestSpec(modalities=modalities)
    frame = load_manifest(config["data"]["manifest"], spec)
    preprocessor = VolumePreprocessor(
        VolumeTransformConfig(
            target_shape=tuple(config["data"]["target_shape"]),
            zscore=bool(config["data"].get("zscore", True)),
        )
    )
    dataset = MissingModalityDataset(filter_split(frame, args.split, spec), spec, preprocessor)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    device = resolve_device(config["training"].get("device", "auto"))
    model = MissingModalityCNN(
        num_modalities=len(modalities),
        num_classes=int(config["model"]["num_classes"]),
        use_modality_mask=bool(config["model"].get("use_modality_mask", True)),
        base_channels=int(config["model"].get("base_channels", 16)),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    metrics = evaluate(
        model,
        loader,
        build_loss(),
        device,
        num_classes=int(config["model"]["num_classes"]),
        desc=args.split,
    )
    print(metrics)
    write_json(save_dir / f"{args.split}_metrics.json", metrics)


if __name__ == "__main__":
    main()
