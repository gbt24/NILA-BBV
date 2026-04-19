from __future__ import annotations

import torch


def evaluate_accuracy(
    model: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).sum().item()
    return float(correct / len(labels))
