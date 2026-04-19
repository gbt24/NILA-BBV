"""Model registry for training baselines."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def _build_mlp(num_classes: int, input_shape: tuple[int, int, int]) -> nn.Module:
    channels, height, width = input_shape
    in_features = channels * height * width
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )


def build_model(
    model_name: str,
    *,
    num_classes: int,
    input_shape: tuple[int, int, int] | None = None,
) -> nn.Module:
    normalized_name = model_name.strip().lower()
    if normalized_name == "resnet18":
        return resnet18(weights=None, num_classes=num_classes)
    if normalized_name == "mlp":
        shape = input_shape if input_shape is not None else (3, 32, 32)
        return _build_mlp(num_classes=num_classes, input_shape=shape)
    raise ValueError(f"unsupported model: {model_name}")
