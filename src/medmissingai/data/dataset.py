from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from medmissingai.data.manifest import ManifestSpec, has_modality
from medmissingai.data.transforms import VolumePreprocessor


def read_nifti(path: str | Path) -> np.ndarray:
    """Read a NIfTI file and return a volume with shape [D, H, W]."""

    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "NIfTI reading requires nibabel. Install project dependencies with "
            "`pip install -e .` or install nibabel directly."
        ) from exc

    nii = nib.load(str(path))
    array = np.asarray(nii.get_fdata(dtype=np.float32))
    if array.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI at {path}, got shape {array.shape}")

    # NiBabel usually returns [X, Y, Z]. PyTorch Conv3d convention below uses
    # [D, H, W], so move the last axis to the front.
    return np.moveaxis(array, -1, 0)


class MissingModalityDataset(Dataset):
    """Dataset for multi-modal 3D classification with missing modalities.

    Returns:
      image: FloatTensor [M, D, H, W]
      modality_mask: FloatTensor [M], 1 if present else 0
      label: LongTensor scalar
      sample_id: str
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        spec: ManifestSpec,
        preprocessor: VolumePreprocessor,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.spec = spec
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        images: list[torch.Tensor] = []
        masks: list[float] = []

        for modality in self.spec.modalities:
            value = row[modality]
            if has_modality(value):
                volume = read_nifti(value)
                image = self.preprocessor(volume)
                mask = 1.0
            else:
                image = torch.zeros(self.preprocessor.config.target_shape, dtype=torch.float32)
                mask = 0.0
            images.append(image)
            masks.append(mask)

        return {
            "image": torch.stack(images, dim=0),
            "modality_mask": torch.tensor(masks, dtype=torch.float32),
            "label": torch.tensor(int(row[self.spec.label_col]), dtype=torch.long),
            "sample_id": str(row[self.spec.sample_id_col]),
        }
