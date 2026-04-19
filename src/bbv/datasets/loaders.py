from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torchvision import datasets


@dataclass(frozen=True)
class LoadedDataset:
    dataset_name: str
    train: bool
    num_classes: int
    num_samples: int
    dataset: datasets.CIFAR10


def load_dataset(root: Path, train: bool, download: bool) -> LoadedDataset:
    dataset = datasets.CIFAR10(root=str(root), train=train, download=download)
    return LoadedDataset(
        dataset_name="cifar10",
        train=train,
        num_classes=len(dataset.classes),
        num_samples=len(dataset),
        dataset=dataset,
    )
