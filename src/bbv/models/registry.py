"""Model registry for training baselines."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def build_model(model_name: str, *, num_classes: int) -> nn.Module:
    normalized_name = model_name.strip().lower()
    if normalized_name == "resnet18":
        return resnet18(weights=None, num_classes=num_classes)
    raise ValueError(f"unsupported model: {model_name}")
