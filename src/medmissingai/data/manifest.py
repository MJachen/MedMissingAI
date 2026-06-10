from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"sample_id", "label", "split"}


@dataclass(frozen=True)
class ManifestSpec:
    """Column contract for a multi-modal imaging manifest.

    Each modality name must match one CSV column. Empty cells mean that the
    modality is missing for that sample.
    """

    modalities: tuple[str, ...]
    sample_id_col: str = "sample_id"
    label_col: str = "label"
    split_col: str = "split"


def load_manifest(path: str | Path, spec: ManifestSpec) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    required = {
        spec.sample_id_col,
        spec.label_col,
        spec.split_col,
        *spec.modalities,
    }
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {missing_columns}")

    if df[spec.sample_id_col].duplicated().any():
        duplicated = df.loc[df[spec.sample_id_col].duplicated(), spec.sample_id_col]
        raise ValueError(f"Duplicated sample_id values: {duplicated.tolist()}")

    allowed_splits = {"train", "val", "test"}
    observed_splits = set(df[spec.split_col].astype(str))
    invalid_splits = sorted(observed_splits.difference(allowed_splits))
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits}")

    return df


def filter_split(df: pd.DataFrame, split: str, spec: ManifestSpec) -> pd.DataFrame:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    subset = df[df[spec.split_col].astype(str) == split].copy()
    subset.reset_index(drop=True, inplace=True)
    return subset


def has_modality(value: object) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text != ""

