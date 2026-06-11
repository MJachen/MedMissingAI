from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from medmissingai.data.dataset import MissingModalityDataset
from medmissingai.data.manifest import ManifestSpec, filter_split, load_manifest
from medmissingai.data.transforms import VolumePreprocessor, VolumeTransformConfig
from medmissingai.models.baseline import MissingModalityCNN
from medmissingai.utils.config import get_save_dir, load_experiment_config
from medmissingai.utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-sample prediction probabilities.")
    parser.add_argument("--config", default=None, help="Optional legacy combined config.")
    parser.add_argument("--data-config", default="configs/data_local.yaml")
    parser.add_argument("--train-config", default="configs/train_local.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None)
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
    output_path = Path(args.output or save_dir / f"predictions_{args.split}.csv")
    checkpoint_path = Path(args.checkpoint or save_dir / "best.pt")

    modalities = tuple(config["data"]["modalities"])
    spec = ManifestSpec(modalities=modalities)
    frame = load_manifest(config["data"]["manifest"], spec)
    split_frame = filter_split(frame, args.split, spec)

    preprocessor = VolumePreprocessor(
        VolumeTransformConfig(
            target_shape=tuple(config["data"]["target_shape"]),
            zscore=bool(config["data"].get("zscore", True)),
        )
    )
    dataset = MissingModalityDataset(split_frame, spec, preprocessor)
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
        use_learnable_missing_token=bool(
            config["model"].get("use_learnable_missing_token", False)
        ),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            modality_mask = batch["modality_mask"].to(device)
            labels = batch["label"].cpu()

            logits = model(image, modality_mask)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)

            batch_size = labels.shape[0]
            batch_frame = split_frame.iloc[offset : offset + batch_size].reset_index(drop=True)
            offset += batch_size

            for i in range(batch_size):
                row: dict[str, Any] = {
                    "sample_id": str(batch["sample_id"][i]),
                    "split": args.split,
                    "label": int(labels[i].item()),
                    "pred_argmax": int(preds[i].item()),
                }
                if "case_id" in batch_frame.columns:
                    row["case_id"] = batch_frame.loc[i, "case_id"]
                if "availability" in batch_frame.columns:
                    row["availability"] = batch_frame.loc[i, "availability"]

                for class_idx, prob in enumerate(probs[i].tolist()):
                    row[f"prob_class_{class_idx}"] = float(prob)
                for modality_idx, modality in enumerate(modalities):
                    row[f"mask_{modality}"] = float(batch["modality_mask"][i, modality_idx].item())
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(f"rows={len(rows)} split={args.split}")


if __name__ == "__main__":
    main()
