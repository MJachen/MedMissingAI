from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from medmissingai.data.modality_settings import DEFAULT_MODALITIES, parse_settings, setting_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BraTS2021 missing-modality manifest.")
    parser.add_argument("--data-root", default=r"E:\brats2021")
    parser.add_argument("--label-file", default=r"E:\EXPS\tensorexps\brats2021_label2020.xlsx")
    parser.add_argument("--output", default="data/brats2021_manifest_smoke.csv")
    parser.add_argument("--settings", default="all")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    return parser.parse_args()


def existing_modal_paths(case_dir: Path, case_id: str) -> dict[str, str]:
    paths = {}
    for modality in DEFAULT_MODALITIES:
        path = case_dir / f"{case_id}_{modality}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing modality file: {path}")
        paths[modality] = str(path)
    return paths


def subset_cases(frame: pd.DataFrame, max_cases: int | None, seed: int) -> pd.DataFrame:
    if max_cases is None or max_cases >= len(frame):
        return frame
    if max_cases < 2:
        raise ValueError("--max-cases must be at least 2")

    _, subset = train_test_split(
        frame,
        test_size=max_cases,
        random_state=seed,
        stratify=frame["label"] if frame["label"].nunique() > 1 else None,
    )
    return subset.sort_values("sample_id").reset_index(drop=True)


def stratify_or_none(frame: pd.DataFrame) -> pd.Series | None:
    if frame["label"].nunique() <= 1:
        return None
    if frame["label"].value_counts().min() < 2:
        return None
    return frame["label"]


def assign_splits(frame: pd.DataFrame, train_frac: float, val_frac: float, seed: int) -> pd.DataFrame:
    if not 0 < train_frac < 1:
        raise ValueError("--train-frac must be in (0,1)")
    if not 0 <= val_frac < 1:
        raise ValueError("--val-frac must be in [0,1)")
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be less than 1")

    train, holdout = train_test_split(
        frame,
        train_size=train_frac,
        random_state=seed,
        stratify=stratify_or_none(frame),
    )
    relative_val = val_frac / (val_frac + test_frac)
    val, test = train_test_split(
        holdout,
        train_size=relative_val,
        random_state=seed,
        stratify=stratify_or_none(holdout),
    )

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")
    return pd.concat([train, val, test], ignore_index=True)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    labels = pd.read_excel(args.label_file)
    labels = labels.rename(columns={"BraTS2021": "sample_id"})
    labels["sample_id"] = labels["sample_id"].astype(str)
    labels["label"] = labels["label"].astype(int)

    rows = []
    for row in labels.itertuples(index=False):
        case_id = row.sample_id
        case_dir = data_root / case_id
        if not case_dir.is_dir():
            continue
        paths = existing_modal_paths(case_dir, case_id)
        rows.append({"sample_id": case_id, "label": int(row.label), **paths})

    case_frame = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    if case_frame.empty:
        raise RuntimeError("No labeled BraTS2021 cases with image directories were found")

    case_frame = subset_cases(case_frame, args.max_cases, args.seed)
    case_frame = assign_splits(case_frame, args.train_frac, args.val_frac, args.seed)
    settings = parse_settings(args.settings, DEFAULT_MODALITIES)

    manifest_rows = []
    for row in case_frame.to_dict(orient="records"):
        for setting in settings:
            available = set(setting)
            output_row = {
                "sample_id": f"{row['sample_id']}__{setting_id(setting)}",
                "case_id": row["sample_id"],
                "label": int(row["label"]),
                "split": row["split"],
                "availability": setting_id(setting),
            }
            for modality in DEFAULT_MODALITIES:
                output_row[modality] = row[modality] if modality in available else ""
            manifest_rows.append(output_row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output, index=False)

    print(f"Wrote {output}")
    print(f"cases={len(case_frame)} rows={len(manifest)} settings={len(settings)}")
    print("case label counts:")
    print(case_frame["label"].value_counts().sort_index().to_string())
    print("manifest split counts:")
    print(manifest["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
