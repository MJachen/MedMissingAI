from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VolumeTransformConfig:
    target_shape: tuple[int, int, int] = (64, 128, 128)
    zscore: bool = True
    eps: float = 1e-6


class VolumePreprocessor:
    """Preprocess a single 3D volume.

    Input shape: [D, H, W].
    Output shape: [D_target, H_target, W_target].
    """

    def __init__(self, config: VolumeTransformConfig):
        self.config = config

    def __call__(self, volume: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(volume, dtype=torch.float32)
        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3D volume [D,H,W], got shape {tuple(tensor.shape)}")

        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if self.config.zscore:
            nonzero = tensor != 0
            if nonzero.any():
                values = tensor[nonzero]
                std = values.std(unbiased=False).clamp_min(self.config.eps)
                tensor = (tensor - values.mean()) / std

        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=self.config.target_shape,
            mode="trilinear",
            align_corners=False,
        )
        return tensor.squeeze(0).squeeze(0)
