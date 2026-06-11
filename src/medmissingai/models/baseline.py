from __future__ import annotations

import torch
from torch import nn


class MissingModalityCNN(nn.Module):
    """Small 3D CNN baseline for missing-modality classification.

    Args:
      num_modalities: number of imaging modalities M.
      num_classes: number of output classes.
      use_modality_mask: if True, concatenate a broadcast modality-mask volume.
      use_learnable_missing_token: if True, replace missing zero volumes with a
        modality-specific learnable scalar before encoding.

    Forward input:
      image: [B, M, D, H, W]
      modality_mask: [B, M]

    Forward output:
      logits: [B, num_classes]
    """

    def __init__(
        self,
        num_modalities: int,
        num_classes: int,
        use_modality_mask: bool = True,
        base_channels: int = 16,
        use_learnable_missing_token: bool = False,
    ) -> None:
        super().__init__()
        in_channels = num_modalities * (2 if use_modality_mask else 1)
        self.num_modalities = num_modalities
        self.use_modality_mask = use_modality_mask
        self.use_learnable_missing_token = use_learnable_missing_token
        if use_learnable_missing_token:
            self.missing_token = nn.Parameter(torch.zeros(num_modalities, 1, 1, 1))
        else:
            self.register_parameter("missing_token", None)

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(base_channels * 4, num_classes)

    def forward(self, image: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"image must have shape [B,M,D,H,W], got {tuple(image.shape)}")
        if modality_mask.ndim != 2:
            raise ValueError(
                f"modality_mask must have shape [B,M], got {tuple(modality_mask.shape)}"
            )

        x = image
        if self.use_learnable_missing_token:
            mask = modality_mask.view(modality_mask.shape[0], self.num_modalities, 1, 1, 1)
            token = self.missing_token.view(1, self.num_modalities, 1, 1, 1)
            x = image * mask + token * (1.0 - mask)

        if self.use_modality_mask:
            bsz, channels, depth, height, width = image.shape
            mask = modality_mask.view(bsz, channels, 1, 1, 1)
            mask = mask.expand(bsz, channels, depth, height, width)
            x = torch.cat([x, mask], dim=1)

        features = self.encoder(x).flatten(1)
        return self.classifier(features)
