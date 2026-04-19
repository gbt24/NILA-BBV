"""Minimal model builders for early-phase smoke runs."""

import torch.nn as nn


def build_simple_classifier(
    input_dim: int, hidden_dim: int, num_classes: int
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )
