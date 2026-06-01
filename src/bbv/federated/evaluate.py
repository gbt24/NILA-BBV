from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def evaluate_accuracy(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
    return float(correct / total) if total > 0 else 0.0
