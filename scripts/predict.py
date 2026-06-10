from __future__ import annotations

import argparse
from pathlib import Path

import torch

from medmissingai.data.dataset import MissingModalityDataset
from medmissingai.data.manifest import ManifestSpec, load_manifest
from medmissingai.data.transforms import VolumePreprocessor, VolumeTransformConfig
from medmissingai.inference.predict import predict_proba
from medmissingai.models.baseline import MissingModalityCNN
from medmissingai.utils.config import get_save_dir, load_experiment_config
from medmissingai.utils.device import resolve_device
from medmissingai.utils.io import write_json
from medmissingai.visualization.heatmap import gradient_saliency, save_middle_slice_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one sample from the manifest.")
    parser.add_argument("--config", default=None, help="Optional legacy combined config.")
    parser.add_argument("--data-config", default="configs/data_local.yaml")
    parser.add_argument("--train-config", default="configs/train_local.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--save-heatmap", action="store_true")
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
    if args.sample_id is not None:
        matches = frame.index[frame[spec.sample_id_col].astype(str) == str(args.sample_id)].tolist()
        if not matches:
            raise ValueError(f"sample_id not found: {args.sample_id}")
        index = matches[0]
    else:
        index = args.index

    preprocessor = VolumePreprocessor(
        VolumeTransformConfig(
            target_shape=tuple(config["data"]["target_shape"]),
            zscore=bool(config["data"].get("zscore", True)),
        )
    )
    dataset = MissingModalityDataset(frame, spec, preprocessor)
    sample = dataset[index]

    device = resolve_device(config["training"].get("device", "auto"))
    model = MissingModalityCNN(
        num_modalities=len(modalities),
        num_classes=int(config["model"]["num_classes"]),
        use_modality_mask=bool(config["model"].get("use_modality_mask", True)),
        base_channels=int(config["model"].get("base_channels", 16)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    probs = predict_proba(model, sample["image"], sample["modality_mask"], device).squeeze(0)
    result = {
        "sample_id": sample["sample_id"],
        "probabilities": {f"class_{i}": float(p) for i, p in enumerate(probs.tolist())},
        "modality_mask": {
            modality: float(sample["modality_mask"][i].item()) for i, modality in enumerate(modalities)
        },
    }

    predictions_dir = save_dir / "predictions"
    write_json(predictions_dir / f"{sample['sample_id']}.json", result)
    print(result)

    if args.save_heatmap:
        heatmap = gradient_saliency(
            model,
            sample["image"],
            sample["modality_mask"],
            target_class=None,
            device=device,
        )
        save_middle_slice_overlay(
            sample["image"],
            heatmap,
            predictions_dir / f"{sample['sample_id']}_heatmap.png",
            modality_index=0,
        )


if __name__ == "__main__":
    main()
