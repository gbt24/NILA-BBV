from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from bbv.datasets.leaf_femnist import load_femnist_split
from bbv.datasets.text import load_sent140_split


class LeafPlaceholderDataset(Dataset):
    def __init__(self, *, num_classes: int, train: bool) -> None:
        size = 64 if train else 16
        self.classes = [str(index) for index in range(num_classes)]
        self.targets = [index % num_classes for index in range(size)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 32, 32), int(self.targets[index])


@dataclass(frozen=True)
class LeafDatasetSpec:
    dataset_name: str
    num_classes: int
    partition_type: str = "natural"
    is_stub: bool = True


@dataclass(frozen=True)
class LeafDatasetStub:
    spec: LeafDatasetSpec
    root: Path
    split_name: str
    download: bool
    dataset: Dataset
    client_indices: list[list[int]] | None = None
    user_ids: list[str] | None = None


_LEAF_DATASETS: dict[str, LeafDatasetSpec] = {
    "femnist": LeafDatasetSpec(dataset_name="femnist", num_classes=62, is_stub=False),
    "shakespeare": LeafDatasetSpec(dataset_name="shakespeare", num_classes=80),
    "sent140": LeafDatasetSpec(dataset_name="sent140", num_classes=2, is_stub=False),
}


def get_leaf_dataset_spec(name: str) -> LeafDatasetSpec:
    normalized_name = name.lower()
    if normalized_name not in _LEAF_DATASETS:
        raise ValueError(f"unsupported LEAF dataset: {name}")
    return _LEAF_DATASETS[normalized_name]


def load_leaf_dataset(
    *, root: Path, train: bool, download: bool, name: str
) -> LeafDatasetStub:
    spec = get_leaf_dataset_spec(name)
    if name.lower() == "femnist":
        split = load_femnist_split(root=root, train=train)
        return LeafDatasetStub(
            spec=spec,
            root=root,
            split_name="train" if train else "test",
            download=download,
            dataset=split.dataset,
            client_indices=split.client_indices,
            user_ids=split.user_ids,
        )
    if name.lower() == "sent140":
        split = load_sent140_split(root=root, train=train)
        return LeafDatasetStub(
            spec=spec,
            root=root,
            split_name="train" if train else "test",
            download=download,
            dataset=split.dataset,
            client_indices=split.client_indices,
            user_ids=split.user_ids,
        )
    return LeafDatasetStub(
        spec=spec,
        root=root,
        split_name="train" if train else "test",
        download=download,
        dataset=LeafPlaceholderDataset(num_classes=spec.num_classes, train=train),
    )
