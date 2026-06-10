from __future__ import annotations

import pandas as pd
import pytest

from medmissingai.data.manifest import ManifestSpec, filter_split, has_modality, load_manifest


def test_manifest_validation_and_split(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"sample_id": "a", "label": 0, "split": "train", "t1": "a_t1.nii.gz", "t2": ""},
            {"sample_id": "b", "label": 1, "split": "test", "t1": "", "t2": "b_t2.nii.gz"},
        ]
    ).to_csv(path, index=False)

    spec = ManifestSpec(modalities=("t1", "t2"))
    frame = load_manifest(path, spec)
    train = filter_split(frame, "train", spec)

    assert len(train) == 1
    assert has_modality(train.loc[0, "t1"])
    assert not has_modality(train.loc[0, "t2"])


def test_manifest_rejects_missing_modality_column(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame([{"sample_id": "a", "label": 0, "split": "train", "t1": "a.nii.gz"}]).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(path, ManifestSpec(modalities=("t1", "t2")))

