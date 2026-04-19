from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PartitionResult:
    method_name: str
    seed: int
    num_clients: int
    total_samples: int
    client_indices: list[list[int]]
    client_sample_counts: list[int]
    client_label_histograms: list[dict[str, int]]


def _validate_partition_inputs(
    labels: Sequence[int], num_clients: int, concentration: float
) -> None:
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0")
    if concentration <= 0:
        raise ValueError("concentration must be greater than 0")
    if len(labels) == 0:
        raise ValueError("labels must not be empty")


def build_partition(
    labels: Sequence[int], *, num_clients: int, concentration: float, seed: int
) -> PartitionResult:
    _validate_partition_inputs(labels, num_clients, concentration)
    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels, dtype=np.int64)
    client_buckets: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in np.unique(label_array):
        class_indices = np.flatnonzero(label_array == class_id)
        shuffled = rng.permutation(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, concentration))
        cut_points = np.cumsum(proportions[:-1]) * len(shuffled)
        splits = np.split(shuffled, cut_points.astype(int))
        for client_index, split in enumerate(splits):
            client_buckets[client_index].extend(int(index) for index in split)

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
        method_name="dirichlet_label_skew",
        seed=seed,
        num_clients=num_clients,
        total_samples=len(labels),
        client_indices=normalized_indices,
        client_sample_counts=[len(indices) for indices in normalized_indices],
        client_label_histograms=histograms,
    )
