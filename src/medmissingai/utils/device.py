from __future__ import annotations

import torch


def resolve_device(value: str | None = None) -> torch.device:
    if value is None or value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)

