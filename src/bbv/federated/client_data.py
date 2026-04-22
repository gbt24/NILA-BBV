from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset, Subset

from bbv.datasets.loaders import LoadedDataset
from bbv.datasets.partitions import PartitionResult


@dataclass(frozen=True)
class ClientDataset:
    client_id: int
    dataset: Dataset
    label_histogram: dict[str, int]


def build_client_datasets(
    loaded_dataset: LoadedDataset, partition: PartitionResult
) -> list[ClientDataset]:
    return [
        ClientDataset(
            client_id=client_id,
            dataset=Subset(loaded_dataset.dataset, indices),
            label_histogram=partition.client_label_histograms[client_id],
        )
        for client_id, indices in enumerate(partition.client_indices)
    ]
