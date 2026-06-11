from __future__ import annotations

import torch

from medmissingai.models.baseline import MissingModalityCNN
from medmissingai.training.engine import apply_modality_dropout
from medmissingai.training.losses import build_loss


def test_balanced_class_weights_use_train_label_counts() -> None:
    criterion = build_loss(class_weights="balanced", labels=[0, 0, 0, 1], num_classes=2)

    assert criterion.weight is not None
    assert torch.allclose(criterion.weight, torch.tensor([2 / 3, 2.0]))


def test_modality_dropout_keeps_at_least_one_present_modality() -> None:
    image = torch.ones(2, 4, 2, 2, 2)
    modality_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])

    dropped_image, dropped_mask = apply_modality_dropout(image, modality_mask, dropout_prob=0.99)

    assert torch.all(dropped_mask.sum(dim=1) >= 1.0)
    assert torch.all(dropped_mask <= modality_mask)
    assert dropped_image.shape == image.shape


def test_missing_token_model_forward_shape() -> None:
    model = MissingModalityCNN(
        num_modalities=4,
        num_classes=2,
        base_channels=2,
        use_learnable_missing_token=True,
    )
    image = torch.randn(2, 4, 8, 8, 8)
    modality_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]])

    logits = model(image, modality_mask)

    assert logits.shape == (2, 2)
