from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bbv.datasets.partitions import PartitionResult


@dataclass(frozen=True)
class SplitMetadata:
    dataset_name: str
    split_method: str
    seed: int
    num_clients: int
    concentration: float
    total_samples: int
    client_sample_counts: list[int]
    client_indices: list[list[int]]
    client_label_histograms: list[dict[str, int]]
    partition_type: str = "dirichlet"


def build_split_metadata(
    dataset_name: str, concentration: float, partition: PartitionResult
) -> SplitMetadata:
    return SplitMetadata(
        dataset_name=dataset_name,
        split_method=partition.method_name,
        seed=partition.seed,
        num_clients=partition.num_clients,
        concentration=concentration,
        total_samples=partition.total_samples,
        client_sample_counts=partition.client_sample_counts,
        client_indices=partition.client_indices,
        client_label_histograms=partition.client_label_histograms,
        partition_type=partition.partition_type,
    )


def save_split_metadata(path: Path, metadata: SplitMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_split_metadata(path: Path) -> SplitMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitMetadata(**payload)
