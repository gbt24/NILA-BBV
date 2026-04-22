"""Model registry for training baselines."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18

from bbv.models.text import build_text_cnn


def _build_mlp(num_classes: int, input_shape: tuple[int, ...]) -> nn.Module:
    in_features = 1
    for dimension in input_shape:
        in_features *= int(dimension)
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
    input_shape: tuple[int, ...] | None = None,
) -> nn.Module:
    normalized_name = model_name.strip().lower()
    if normalized_name == "resnet18":
        return resnet18(weights=None, num_classes=num_classes)
    if normalized_name == "mlp":
        shape = input_shape if input_shape is not None else (3, 32, 32)
        return _build_mlp(num_classes=num_classes, input_shape=shape)
    if normalized_name == "text_cnn":
        return build_text_cnn(num_classes=num_classes, input_shape=input_shape)
    raise ValueError(f"unsupported model: {model_name}")
