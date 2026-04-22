from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets

from bbv.datasets.leaf import get_leaf_dataset_spec, load_leaf_dataset
from bbv.datasets.transforms import build_image_transform


@dataclass(frozen=True)
class LoadedDataset:
    dataset_name: str
    split_name: str
    train: bool
    num_classes: int
    num_samples: int
    dataset: Dataset
    metadata: dict[str, object] = field(default_factory=dict)


_VISION_DATASETS = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
}

_LEAF_DATASETS = {"femnist", "shakespeare", "sent140"}


def load_dataset(
    root: Path, train: bool, download: bool, name: str = "cifar10"
) -> LoadedDataset:
    normalized_name = name.lower()
    if normalized_name in _LEAF_DATASETS:
        leaf_dataset = load_leaf_dataset(
            root=root,
            train=train,
            download=download,
            name=normalized_name,
        )
        spec = get_leaf_dataset_spec(normalized_name)
        return LoadedDataset(
            dataset_name=normalized_name,
            split_name=leaf_dataset.split_name,
            train=train,
            num_classes=spec.num_classes,
            num_samples=len(leaf_dataset.dataset),
            dataset=leaf_dataset.dataset,
            metadata={
                "dataset_name": normalized_name,
                "partition_type": spec.partition_type,
                "is_stub": spec.is_stub,
                "natural_client_indices": getattr(leaf_dataset, "client_indices", None),
                "user_ids": getattr(leaf_dataset, "user_ids", None),
            },
        )

    if normalized_name not in _VISION_DATASETS:
        raise ValueError(f"unsupported dataset: {name}")

    transform = build_image_transform(train=train)
    dataset_cls = getattr(datasets, _VISION_DATASETS[normalized_name])
    try:
        dataset = dataset_cls(
            root=str(root), train=train, download=download, transform=transform
        )
    except TypeError:
        dataset = dataset_cls(root=str(root), train=train, download=download)

    return LoadedDataset(
        dataset_name=normalized_name,
        split_name="train" if train else "test",
        train=train,
        num_classes=len(dataset.classes),
        num_samples=len(dataset),
        dataset=dataset,
        metadata={
            "dataset_name": normalized_name,
            "partition_type": "dirichlet",
            "is_stub": False,
        },
    )
