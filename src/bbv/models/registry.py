"""Model registry for training baselines."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock, ResNet

from bbv.models.text import build_text_cnn


def _build_cifar_resnet18(num_classes: int) -> nn.Module:
    """ResNet-18 adapted for 32x32 CIFAR images.

    Replaces the standard 7x7 stride-2 conv1 with 3x3 stride-1 conv1 and
    removes the initial maxpool, preserving spatial resolution for small inputs.
    The default torchvision ResNet-18 is designed for 224x224 ImageNet images;
    on 32x32 CIFAR-10 its aggressive early downsampling (32->15 through conv1,
    then ->7 through maxpool) discards critical fine-grained detail.
    """
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the initial maxpool (set to identity)
    model.maxpool = nn.Identity()
    return model


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
    if normalized_name == "resnet18_cifar":
        return _build_cifar_resnet18(num_classes=num_classes)
    if normalized_name == "mlp":
        shape = input_shape if input_shape is not None else (3, 32, 32)
        return _build_mlp(num_classes=num_classes, input_shape=shape)
    if normalized_name == "text_cnn":
        return build_text_cnn(num_classes=num_classes, input_shape=input_shape)
    raise ValueError(f"unsupported model: {model_name}")
