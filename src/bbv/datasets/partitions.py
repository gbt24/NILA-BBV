from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PartitionResult:
    method_name: str
    partition_type: str
    partition_params: dict[str, float | int | str]
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


def _sample_quantity_counts(
    *, total_samples: int, num_clients: int, quantity_sigma: float, seed: int
) -> np.ndarray:
    if quantity_sigma < 0:
        raise ValueError("quantity_sigma must be greater than or equal to 0")

    rng = np.random.default_rng(seed)
    if quantity_sigma == 0:
        weights = np.ones(num_clients, dtype=np.float64)
    else:
        weights = rng.lognormal(mean=0.0, sigma=quantity_sigma, size=num_clients)
    proportions = weights / weights.sum()
    counts = np.floor(proportions * total_samples).astype(int)
    remainder = total_samples - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(proportions * total_samples - counts), kind="stable")
        for index in order[:remainder]:
            counts[index] += 1
    return counts


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


def _build_quantity_skew_partition(
    label_array: np.ndarray, num_clients: int, quantity_sigma: float, seed: int
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(label_array))
    counts = _sample_quantity_counts(
        total_samples=len(label_array),
        num_clients=num_clients,
        quantity_sigma=quantity_sigma,
        seed=seed,
    )

    client_buckets: list[list[int]] = []
    start = 0
    for count in counts:
        stop = start + int(count)
        client_buckets.append([int(index) for index in shuffled_indices[start:stop]])
        start = stop
    return client_buckets


def _build_combined_partition(
    label_array: np.ndarray,
    num_clients: int,
    concentration: float,
    quantity_sigma: float,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    target_counts = _sample_quantity_counts(
        total_samples=len(label_array),
        num_clients=num_clients,
        quantity_sigma=quantity_sigma,
        seed=seed,
    )
    remaining_capacity = target_counts.astype(int).copy()
    client_buckets: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in np.unique(label_array):
        class_indices = rng.permutation(np.flatnonzero(label_array == class_id))
        if len(class_indices) == 0:
            continue

        base = rng.dirichlet(np.full(num_clients, concentration))
        weighted = base * np.maximum(remaining_capacity, 1)
        if weighted.sum() == 0:
            weighted = np.maximum(remaining_capacity, 1)
        class_counts = np.floor(weighted / weighted.sum() * len(class_indices)).astype(int)
        remainder = len(class_indices) - int(class_counts.sum())
        if remainder > 0:
            order = np.argsort(-weighted, kind="stable")
            for index in order[:remainder]:
                class_counts[index] += 1

        cursor = 0
        for client_index, requested in enumerate(class_counts):
            take = min(int(requested), int(remaining_capacity[client_index]))
            if take <= 0:
                continue
            selected = class_indices[cursor : cursor + take]
            client_buckets[client_index].extend(int(index) for index in selected)
            remaining_capacity[client_index] -= take
            cursor += take

        leftovers = class_indices[cursor:]
        if len(leftovers) == 0:
            continue
        for index in leftovers:
            available = np.flatnonzero(remaining_capacity > 0)
            if len(available) == 0:
                available = np.arange(num_clients)
            chosen = int(rng.choice(available))
            client_buckets[chosen].append(int(index))
            if remaining_capacity[chosen] > 0:
                remaining_capacity[chosen] -= 1

    return client_buckets


def _build_natural_partition(
    label_array: np.ndarray, num_clients: int
) -> list[list[int]]:
    ordered_indices = np.arange(len(label_array), dtype=np.int64)
    splits = np.array_split(ordered_indices, num_clients)
    return [[int(index) for index in split] for split in splits]


def _rebalance_empty_clients(
    client_buckets: list[list[int]], total_samples: int
) -> list[list[int]]:
    if total_samples < len(client_buckets):
        return client_buckets

    for client_index, bucket in enumerate(client_buckets):
        if bucket:
            continue
        donor_index = max(range(len(client_buckets)), key=lambda idx: len(client_buckets[idx]))
        donor_bucket = client_buckets[donor_index]
        if len(donor_bucket) <= 1:
            break
        bucket.append(donor_bucket.pop())
    return client_buckets


def build_partition(
    labels: Sequence[int],
    *,
    num_clients: int,
    concentration: float,
    seed: int,
    partition_type: str = "dirichlet",
    shards_per_client: int = 2,
    quantity_sigma: float = 0.0,
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
        partition_params: dict[str, float | int | str] = {"concentration": concentration}
    elif partition_type == "shard":
        client_buckets = _build_shard_partition(
            label_array=label_array,
            num_clients=num_clients,
            shards_per_client=shards_per_client,
            seed=seed,
        )
        method_name = "shard"
        partition_params = {"concentration": concentration, "shards_per_client": shards_per_client}
    elif partition_type == "quantity_skew":
        client_buckets = _build_quantity_skew_partition(
            label_array=label_array,
            num_clients=num_clients,
            quantity_sigma=quantity_sigma,
            seed=seed,
        )
        method_name = "quantity_skew"
        partition_params = {"concentration": concentration, "quantity_sigma": quantity_sigma}
    elif partition_type == "combined_label_quantity":
        client_buckets = _build_combined_partition(
            label_array=label_array,
            num_clients=num_clients,
            concentration=concentration,
            quantity_sigma=quantity_sigma,
            seed=seed,
        )
        method_name = "combined_label_quantity"
        partition_params = {"concentration": concentration, "quantity_sigma": quantity_sigma}
    elif partition_type == "natural":
        client_buckets = _build_natural_partition(
            label_array=label_array,
            num_clients=num_clients,
        )
        method_name = "natural"
        partition_params = {}
    else:
        raise ValueError(f"unsupported partition_type: {partition_type}")

    client_buckets = _rebalance_empty_clients(client_buckets, len(labels))

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
        partition_params=partition_params,
        seed=seed,
        num_clients=num_clients,
        total_samples=len(labels),
        client_indices=normalized_indices,
        client_sample_counts=[len(indices) for indices in normalized_indices],
        client_label_histograms=histograms,
    )
