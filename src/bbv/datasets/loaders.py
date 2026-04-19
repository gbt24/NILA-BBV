from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torchvision import datasets

from bbv.datasets.transforms import build_image_transform


@dataclass(frozen=True)
class LoadedDataset:
    dataset_name: str
    split_name: str
    train: bool
    num_classes: int
    num_samples: int
    dataset: datasets.CIFAR10


def load_dataset(
    root: Path, train: bool, download: bool, name: str = "cifar10"
) -> LoadedDataset:
    if name != "cifar10":
        raise ValueError(f"unsupported dataset: {name}")

    transform = build_image_transform(train=train)
    try:
        dataset = datasets.CIFAR10(
            root=str(root), train=train, download=download, transform=transform
        )
    except TypeError:
        dataset = datasets.CIFAR10(root=str(root), train=train, download=download)

    return LoadedDataset(
        dataset_name="cifar10",
        split_name="train" if train else "test",
        train=train,
        num_classes=len(dataset.classes),
        num_samples=len(dataset),
        dataset=dataset,
    )
