from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PartitionResult:
    method_name: str
    partition_type: str
    seed: int
    num_clients: int
    total_samples: int
    client_indices: list[list[int]]
    client_sample_counts: list[int]
    client_label_histograms: list[dict[str, int]]


def _validate_partition_inputs(
    labels: Sequence[int], num_clients: int, concentration: float, shards_per_client: int
) -> None:
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0")
    if concentration <= 0:
        raise ValueError("concentration must be greater than 0")
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
    if shards_per_client <= 0:
        raise ValueError("shards_per_client must be greater than 0")


def _build_dirichlet_partition(
    label_array: np.ndarray, num_clients: int, concentration: float, seed: int
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    client_buckets: list[list[int]] = [[] for _ in range(num_clients)]
    for class_id in np.unique(label_array):
        class_indices = np.flatnonzero(label_array == class_id)
        shuffled = rng.permutation(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, concentration))
        cut_points = np.cumsum(proportions[:-1]) * len(shuffled)
        splits = np.split(shuffled, cut_points.astype(int))
        for client_index, split in enumerate(splits):
            client_buckets[client_index].extend(int(index) for index in split)
    return client_buckets


def _build_shard_partition(
    label_array: np.ndarray, num_clients: int, shards_per_client: int, seed: int
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    sorted_indices = np.argsort(label_array, kind="stable")
    total_shards = num_clients * shards_per_client
    shards = np.array_split(sorted_indices, total_shards)
    shard_order = rng.permutation(total_shards)
    client_buckets: list[list[int]] = [[] for _ in range(num_clients)]
    for client_index in range(num_clients):
        shard_ids = shard_order[
            client_index * shards_per_client : (client_index + 1) * shards_per_client
        ]
        for shard_id in shard_ids:
            client_buckets[client_index].extend(int(index) for index in shards[shard_id])
    return client_buckets


def build_partition(
    labels: Sequence[int],
    *,
    num_clients: int,
    concentration: float,
    seed: int,
    partition_type: str = "dirichlet",
    shards_per_client: int = 2,
) -> PartitionResult:
    _validate_partition_inputs(labels, num_clients, concentration, shards_per_client)
    label_array = np.asarray(labels, dtype=np.int64)
    if partition_type == "dirichlet":
        client_buckets = _build_dirichlet_partition(
            label_array=label_array,
            num_clients=num_clients,
            concentration=concentration,
            seed=seed,
        )
        method_name = "dirichlet_label_skew"
    elif partition_type == "shard":
        client_buckets = _build_shard_partition(
            label_array=label_array,
            num_clients=num_clients,
            shards_per_client=shards_per_client,
            seed=seed,
        )
        method_name = "shard"
    else:
        raise ValueError(f"unsupported partition_type: {partition_type}")

    normalized_indices = [sorted(indices) for indices in client_buckets]
    flattened = sorted(index for indices in normalized_indices for index in indices)
    expected = list(range(len(labels)))
    if flattened != expected:
        raise ValueError("partition must cover every sample exactly once")

    histograms: list[dict[str, int]] = []
    for indices in normalized_indices:
        counts: dict[str, int] = {}
        for index in indices:
            label = str(int(label_array[index]))
            counts[label] = counts.get(label, 0) + 1
        histograms.append(counts)

    return PartitionResult(
        method_name=method_name,
        partition_type=partition_type,
        seed=seed,
        num_clients=num_clients,
        total_samples=len(labels),
        client_indices=normalized_indices,
        client_sample_counts=[len(indices) for indices in normalized_indices],
        client_label_histograms=histograms,
    )
