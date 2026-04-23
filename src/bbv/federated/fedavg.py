"""Minimal FedAvg baseline pipeline for Phase 2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from bbv.allocation import (
    allocate_watermark_budget,
    build_client_stats,
    build_client_stats_from_histogram,
    estimate_adaptability,
)
from bbv.datasets.loaders import load_dataset
from bbv.datasets.metadata import build_split_metadata
from bbv.datasets.partitions import PartitionResult, build_partition
from bbv.federated.evaluate import evaluate_accuracy
from bbv.federated.client_data import ClientDataset, build_client_datasets
from bbv.federated.hooks import WatermarkHook
from bbv.federated.progress import progress_iterable
from bbv.models import build_model
from bbv.utils.io import write_json
from bbv.watermarking import compute_loss_components
from bbv.watermarking.baseline import save_owner_artifacts


@dataclass(frozen=True)
class FederatedClient:
    client_id: int
    dataset: ClientDataset
    labels: torch.Tensor


@dataclass(frozen=True)
class FederatedServer:
    model: torch.nn.Module


@dataclass(frozen=True)
class FedAvgResult:
    run_dir: Path
    metrics_path: Path
    metadata_path: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    allocation_path: Path | None

    @property
    def output_dir(self) -> Path:
        return self.run_dir


def build_model_input_shape(dataset_name: str) -> tuple[int, ...]:
    normalized_name = dataset_name.lower()
    if normalized_name not in {"cifar10", "cifar100", "femnist", "shakespeare", "sent140"}:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    if normalized_name in {"sent140", "shakespeare"}:
        return (32,)
    return (3, 32, 32)


def build_client(
    *,
    client_id: int,
    dataset_name: str,
    samples_per_client: int,
    num_classes: int,
    seed: int,
) -> FederatedClient:
    input_shape = build_model_input_shape(dataset_name)
    generator = torch.Generator().manual_seed(seed + client_id)
    if len(input_shape) == 1:
        sequence_length = input_shape[0]
        features = torch.randint(
            low=0,
            high=2048,
            size=(samples_per_client, sequence_length),
            generator=generator,
            dtype=torch.long,
        )
    else:
        channels, height, width = input_shape
        features = torch.randn(
            samples_per_client,
            channels,
            height,
            width,
            generator=generator,
        )
    labels = torch.randint(
        low=0,
        high=num_classes,
        size=(samples_per_client,),
        generator=generator,
    )
    return FederatedClient(
        client_id=client_id,
        dataset=ClientDataset(
            client_id=client_id,
            dataset=TensorDataset(features, labels),
            label_histogram=_build_label_histogram(labels),
        ),
        labels=labels,
    )


def build_server(
    *, model_name: str, num_classes: int, seed: int, input_shape: tuple[int, ...] | None = None
) -> FederatedServer:
    torch.manual_seed(seed)
    return FederatedServer(
        model=build_model(
            model_name,
            num_classes=num_classes,
            input_shape=input_shape if input_shape is not None else build_model_input_shape("cifar10"),
        )
    )


def _train_one_client(
    *,
    global_model: torch.nn.Module,
    client: FederatedClient,
    learning_rate: float,
    local_epochs: int,
    batch_size: int,
    watermark_hook: WatermarkHook | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    local_model = deepcopy(global_model)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    data_loader = DataLoader(
        client.dataset.dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    running_task_losses: list[float] = []
    running_wm_losses: list[float] = []
    running_total_losses: list[float] = []
    for _ in range(local_epochs):
        for features, labels in data_loader:
            optimizer.zero_grad()
            logits = local_model(features)
            task_loss = F.cross_entropy(logits, labels)
            wm_loss = torch.zeros((), dtype=task_loss.dtype)
            total_loss = task_loss
            if watermark_hook is not None:
                query_batch = watermark_hook.build_query_batch(features.shape[1:]).to(
                    device=features.device,
                    dtype=features.dtype,
                )
                query_logits = local_model(query_batch)
                query_bits = torch.tensor(
                    watermark_hook.codebook,
                    dtype=torch.long,
                    device=query_logits.device,
                )
                total_loss, task_loss, wm_loss = compute_loss_components(
                    logits_main=logits,
                    labels_main=labels,
                    logits_query=query_logits,
                    bits=query_bits,
                    task_weight=1.0,
                    wm_weight=watermark_hook.wm_weight,
                )
            total_loss.backward()
            optimizer.step()
            running_task_losses.append(float(task_loss.detach().item()))
            running_wm_losses.append(float(wm_loss.detach().item()))
            running_total_losses.append(float(total_loss.detach().item()))
    avg_losses = {
        "task_loss": sum(running_task_losses) / len(running_task_losses),
        "wm_loss": sum(running_wm_losses) / len(running_wm_losses),
        "total_loss": sum(running_total_losses) / len(running_total_losses),
    }
    return local_model.state_dict(), avg_losses


def _average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        stacked = torch.stack([state[key] for state in state_dicts], dim=0)
        if torch.is_floating_point(stacked):
            averaged[key] = stacked.mean(dim=0)
        else:
            averaged[key] = stacked[0]
    return averaged


def _create_run_dir(output_root: Path) -> Path:
    run_id = f"fedavg-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _validate_inputs(
    num_clients: int,
    rounds: int,
    participation_rate: float,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    samples_per_client: int,
    allocation_budget_ratio: float,
    allocation_base_loss_weight: float,
) -> None:
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0")
    if rounds <= 0:
        raise ValueError("rounds must be greater than 0")
    if participation_rate <= 0.0 or participation_rate > 1.0:
        raise ValueError("participation_rate must be in (0, 1]")
    if local_epochs <= 0:
        raise ValueError("local_epochs must be greater than 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0")
    if samples_per_client <= 0:
        raise ValueError("samples_per_client must be greater than 0")
    if allocation_budget_ratio < 0.0 or allocation_budget_ratio > 1.0:
        raise ValueError("allocation_budget_ratio must be in [0, 1]")
    if allocation_base_loss_weight < 0.0:
        raise ValueError("allocation_base_loss_weight must be non-negative")


def _build_label_histogram(labels: torch.Tensor) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for label in labels.tolist():
        key = str(int(label))
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _build_labels_tensor(client_dataset: ClientDataset) -> torch.Tensor:
    labels: list[int] = []
    for _, label in client_dataset.dataset:
        labels.append(int(label))
    return torch.tensor(labels, dtype=torch.long)


def _build_partition_from_natural_clients(
    *,
    labels: list[int],
    natural_client_indices: list[list[int]],
    num_clients: int,
    samples_per_client: int,
    seed: int,
) -> tuple[PartitionResult, list[int]]:
    eligible_client_ids = [
        client_id
        for client_id, indices in enumerate(natural_client_indices)
        if len(indices) >= samples_per_client
    ]
    if len(eligible_client_ids) < num_clients:
        raise ValueError("natural split does not provide enough clients with sufficient samples")

    generator = torch.Generator().manual_seed(seed)
    sampled_positions = torch.randperm(len(eligible_client_ids), generator=generator).tolist()[:num_clients]
    selected_client_ids = [eligible_client_ids[position] for position in sampled_positions]
    sampled_indices: list[int] = []
    client_indices: list[list[int]] = []
    histograms: list[dict[str, int]] = []

    for client_id in selected_client_ids:
        raw_indices = sorted(natural_client_indices[client_id])
        if len(raw_indices) < samples_per_client:
            raise ValueError(
                "samples_per_client exceeds available samples for a natural-split client"
            )
        selected_indices = raw_indices[:samples_per_client]
        client_indices.append(selected_indices)
        sampled_indices.extend(selected_indices)

        histogram: dict[str, int] = {}
        for index in selected_indices:
            label = str(int(labels[index]))
            histogram[label] = histogram.get(label, 0) + 1
        histograms.append(histogram)

    partition = PartitionResult(
        method_name="natural",
        partition_type="natural",
        partition_params={"source": "natural_client_indices"},
        seed=seed,
        num_clients=num_clients,
        total_samples=len(sampled_indices),
        client_indices=client_indices,
        client_sample_counts=[len(indices) for indices in client_indices],
        client_label_histograms=histograms,
    )
    return partition, sampled_indices


def _flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            parts.append(parameter.grad.detach().flatten())
    if not parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(parts)


def _compute_client_allocation_stats(
    *,
    model: torch.nn.Module,
    client: FederatedClient,
    batch_size: int,
    num_classes: int,
    watermark_hook: WatermarkHook | None,
) -> object:
    stats_model = deepcopy(model)
    stats_model.train()
    data_loader = DataLoader(client.dataset.dataset, batch_size=batch_size, shuffle=False)
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for features, labels in data_loader:
        feature_batches.append(features)
        label_batches.append(labels)
    features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)

    stats_model.zero_grad()
    main_logits = stats_model(features)
    F.cross_entropy(main_logits, labels).backward()
    main_gradient = _flatten_gradients(stats_model)

    stats_model.zero_grad()
    if watermark_hook is not None:
        query_batch = watermark_hook.build_query_batch(features.shape[1:]).to(
            dtype=features.dtype,
            device=features.device,
        )
        wm_logits = stats_model(query_batch)
        wm_labels = torch.tensor(
            watermark_hook.codebook,
            dtype=torch.long,
            device=features.device,
        )
        _, _, wm_loss = compute_loss_components(
            logits_main=main_logits.detach(),
            labels_main=labels,
            logits_query=wm_logits,
            bits=wm_labels,
            task_weight=1.0,
            wm_weight=1.0,
        )
        wm_loss.backward()
    else:
        wm_logits = stats_model(features)
        wm_labels = (labels + 1) % max(num_classes, 2)
        F.cross_entropy(wm_logits, wm_labels).backward()
    wm_gradient = _flatten_gradients(stats_model)

    return build_client_stats(
        labels=client.labels,
        main_gradient=main_gradient,
        wm_gradient=wm_gradient,
        privacy_penalty=0.1,
    )


def _enforce_client_sample_budget(
    *,
    partition: PartitionResult,
    labels: list[int],
    samples_per_client: int,
) -> PartitionResult:
    target_total = partition.num_clients * samples_per_client
    if len(labels) < target_total:
        raise ValueError(
            "samples_per_client requires at least num_clients * samples_per_client "
            "dataset samples"
        )

    client_indices = [list(indices) for indices in partition.client_indices]
    overflow: list[int] = []
    for indices in client_indices:
        while len(indices) > samples_per_client:
            overflow.append(indices.pop())

    for indices in client_indices:
        while len(indices) < samples_per_client:
            indices.append(overflow.pop())

    normalized_indices = [sorted(indices) for indices in client_indices]
    histograms: list[dict[str, int]] = []
    for indices in normalized_indices:
        counts: dict[str, int] = {}
        for index in indices:
            label = str(int(labels[index]))
            counts[label] = counts.get(label, 0) + 1
        histograms.append(counts)

    return PartitionResult(
        method_name=partition.method_name,
        partition_type=partition.partition_type,
        partition_params=partition.partition_params,
        seed=partition.seed,
        num_clients=partition.num_clients,
        total_samples=len(labels),
        client_indices=normalized_indices,
        client_sample_counts=[len(indices) for indices in normalized_indices],
        client_label_histograms=histograms,
    )


def _enforce_minimum_client_samples(
    *, partition: PartitionResult, labels: list[int], minimum_samples: int
) -> PartitionResult:
    if minimum_samples <= 1 or len(labels) < partition.num_clients * minimum_samples:
        return partition

    client_indices = [list(indices) for indices in partition.client_indices]
    while True:
        receiver_index = min(range(len(client_indices)), key=lambda idx: len(client_indices[idx]))
        donor_index = max(range(len(client_indices)), key=lambda idx: len(client_indices[idx]))
        if len(client_indices[receiver_index]) >= minimum_samples:
            break
        if len(client_indices[donor_index]) <= minimum_samples:
            break
        client_indices[receiver_index].append(client_indices[donor_index].pop())

    normalized_indices = [sorted(indices) for indices in client_indices]
    histograms: list[dict[str, int]] = []
    for indices in normalized_indices:
        counts: dict[str, int] = {}
        for index in indices:
            label = str(int(labels[index]))
            counts[label] = counts.get(label, 0) + 1
        histograms.append(counts)

    return PartitionResult(
        method_name=partition.method_name,
        partition_type=partition.partition_type,
        partition_params=partition.partition_params,
        seed=partition.seed,
        num_clients=partition.num_clients,
        total_samples=partition.total_samples,
        client_indices=normalized_indices,
        client_sample_counts=[len(indices) for indices in normalized_indices],
        client_label_histograms=histograms,
    )


def _remap_partition_indices(
    *, partition: PartitionResult, sampled_indices: list[int], labels: list[int]
) -> PartitionResult:
    remapped_indices = [
        [sampled_indices[index] for index in client_indices]
        for client_indices in partition.client_indices
    ]
    normalized_indices = [sorted(indices) for indices in remapped_indices]
    histograms: list[dict[str, int]] = []
    for indices in normalized_indices:
        counts: dict[str, int] = {}
        for index in indices:
            label = str(int(labels[index]))
            counts[label] = counts.get(label, 0) + 1
        histograms.append(counts)

    return PartitionResult(
        method_name=partition.method_name,
        partition_type=partition.partition_type,
        partition_params=partition.partition_params,
        seed=partition.seed,
        num_clients=partition.num_clients,
        total_samples=partition.total_samples,
        client_indices=normalized_indices,
        client_sample_counts=[len(indices) for indices in normalized_indices],
        client_label_histograms=histograms,
    )


def _build_evaluation_tensors(
    dataset: object, max_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[int] = []
    for index in range(min(len(dataset), max_samples)):
        feature, label = dataset[index]
        features.append(feature)
        labels.append(int(label))
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def _infer_input_shape(dataset: object) -> tuple[int, ...]:
    sample, _ = dataset[0]
    return tuple(int(dimension) for dimension in sample.shape)


def train_federated(
    *,
    output_root: Path,
    seed: int,
    dataset_name: str,
    model_name: str,
    num_classes: int,
    num_clients: int,
    rounds: int,
    participation_rate: float,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    samples_per_client: int,
    partition_type: str = "dirichlet",
    concentration: float = 1.0,
    shards_per_client: int = 2,
    quantity_sigma: float = 0.0,
    watermark_hook: WatermarkHook | None = None,
    allocation_enabled: bool = False,
    allocation_budget_ratio: float = 0.3,
    allocation_base_loss_weight: float = 0.1,
    progress_enabled: bool = True,
) -> FedAvgResult:
    _validate_inputs(
        num_clients=num_clients,
        rounds=rounds,
        participation_rate=participation_rate,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        samples_per_client=samples_per_client,
        allocation_budget_ratio=allocation_budget_ratio,
        allocation_base_loss_weight=allocation_base_loss_weight,
    )
    torch.manual_seed(seed)
    run_dir = _create_run_dir(Path(output_root))
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "run_metadata.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    allocation_path = run_dir / "allocation_assignments.json"

    loaded_dataset = load_dataset(
        name=dataset_name,
        root=Path("data/raw"),
        train=True,
        download=True,
    )
    validation_dataset = load_dataset(
        name=dataset_name,
        root=Path("data/raw"),
        train=False,
        download=True,
    )
    input_shape = _infer_input_shape(loaded_dataset.dataset)
    val_features, val_labels = _build_evaluation_tensors(validation_dataset.dataset, 64)
    labels = [int(label) for label in getattr(loaded_dataset.dataset, "targets")]
    dataset_metadata = getattr(loaded_dataset, "metadata", {}) or {}
    natural_client_indices = dataset_metadata.get("natural_client_indices")
    if partition_type == "natural" and natural_client_indices is not None:
        dataset_partition, sampled_indices = _build_partition_from_natural_clients(
            labels=labels,
            natural_client_indices=[list(indices) for indices in natural_client_indices],
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            seed=seed,
        )
        partition = dataset_partition
    elif partition_type == "natural":
        sampled_indices = list(range(len(labels)))
        selected_labels = [labels[index] for index in sampled_indices]
        partition = build_partition(
            selected_labels,
            num_clients=num_clients,
            concentration=concentration,
            seed=seed,
            partition_type=partition_type,
            shards_per_client=shards_per_client,
            quantity_sigma=quantity_sigma,
        )
        dataset_partition = _remap_partition_indices(
            partition=partition,
            sampled_indices=sampled_indices,
            labels=labels,
        )
    else:
        max_samples = num_clients * samples_per_client
        if len(labels) < max_samples:
            raise ValueError(
                "samples_per_client requires at least num_clients * samples_per_client "
                "dataset samples"
            )
        sample_generator = torch.Generator().manual_seed(seed)
        sampled_indices = torch.randperm(len(labels), generator=sample_generator).tolist()[:max_samples]
        selected_labels = [labels[index] for index in sampled_indices]
        partition = build_partition(
            selected_labels,
            num_clients=num_clients,
            concentration=concentration,
            seed=seed,
            partition_type=partition_type,
            shards_per_client=shards_per_client,
            quantity_sigma=quantity_sigma,
        )
        if partition_type in {"dirichlet", "shard"}:
            partition = _enforce_client_sample_budget(
                partition=partition,
                labels=selected_labels,
                samples_per_client=samples_per_client,
            )
        elif partition_type in {"quantity_skew", "combined_label_quantity"}:
            partition = _enforce_minimum_client_samples(
                partition=partition,
                labels=selected_labels,
                minimum_samples=2,
            )
        dataset_partition = _remap_partition_indices(
            partition=partition,
            sampled_indices=sampled_indices,
            labels=labels,
        )
    partition_metadata = build_split_metadata(
        loaded_dataset.dataset_name,
        concentration=concentration,
        partition=partition,
    )
    client_datasets = build_client_datasets(loaded_dataset, dataset_partition)
    clients = [
        FederatedClient(
            client_id=client_dataset.client_id,
            dataset=client_dataset,
            labels=_build_labels_tensor(client_dataset),
        )
        for client_dataset in client_datasets
    ]
    server = build_server(
        model_name=model_name,
        num_classes=num_classes,
        seed=seed,
        input_shape=input_shape,
    )

    selected_per_round = max(1, int(round(num_clients * participation_rate)))
    round_metrics: list[dict[str, float | int]] = []
    round_assignments: list[dict[str, object]] = []
    rng = torch.Generator().manual_seed(seed)
    best_val_accuracy = -1.0

    round_iterator = progress_iterable(
        range(1, rounds + 1),
        description="Rounds",
        enabled=progress_enabled,
        leave=True,
        position=0,
    )
    for round_id in round_iterator:
        permutation = torch.randperm(num_clients, generator=rng).tolist()
        selected_ids = permutation[:selected_per_round]
        selected_clients = [clients[client_id] for client_id in selected_ids]

        assignment_for_round: dict[int, dict[str, float | int | bool]] = {}
        if allocation_enabled:
            selected_histograms = [
                _build_label_histogram(client.labels) for client in selected_clients
            ]
            selected_stats = [
                _compute_client_allocation_stats(
                    model=server.model,
                    client=client,
                    batch_size=batch_size,
                    num_classes=num_classes,
                    watermark_hook=watermark_hook,
                )
                for client in selected_clients
            ]
            adaptability_scores = estimate_adaptability(
                stats=selected_stats
            )
            allocation_budget_clients = int(round(len(selected_clients) * allocation_budget_ratio))
            if allocation_budget_ratio > 0.0 and allocation_budget_clients == 0:
                allocation_budget_clients = 1
            if allocation_budget_clients > 0:
                assignment_local_ids = allocate_watermark_budget(
                    adaptability_scores=adaptability_scores,
                    budget_clients=allocation_budget_clients,
                    base_loss_weight=allocation_base_loss_weight,
                )
            else:
                assignment_local_ids = {
                    local_id: {
                        "enabled": False,
                        "loss_weight": 0.0,
                        "depth": 0,
                        "score": float(adaptability_scores[local_id]),
                    }
                    for local_id in adaptability_scores
                }
            assignment_for_round = {
                selected_ids[local_id]: {
                    **assignment,
                    "stats": {
                        "class_coverage": selected_stats[local_id].class_coverage,
                        "skew_ratio": selected_stats[local_id].skew_ratio,
                        "main_wm_alignment": selected_stats[local_id].main_wm_alignment,
                        "privacy_penalty": selected_stats[local_id].privacy_penalty,
                    },
                }
                for local_id, assignment in assignment_local_ids.items()
            }
            round_assignments.append(
                {
                    "round": round_id,
                    "selected_client_ids": selected_ids,
                    "selected_histograms": selected_histograms,
                    "selected_histograms_by_client": {
                        str(client_id): histogram
                        for client_id, histogram in zip(selected_ids, selected_histograms, strict=False)
                    },
                    "assignments": assignment_for_round,
                }
            )

        local_states: list[dict[str, torch.Tensor]] = []
        local_loss_summaries: list[dict[str, float]] = []
        client_iterator = progress_iterable(
            selected_clients,
            description=f"Clients (round {round_id})",
            enabled=progress_enabled,
            leave=False,
            position=1,
        )
        for client in client_iterator:
            client_watermark_hook = watermark_hook
            if watermark_hook is not None and allocation_enabled:
                assignment = assignment_for_round.get(client.client_id, {"enabled": False, "loss_weight": 0.0})
                if not bool(assignment["enabled"]):
                    client_watermark_hook = None
                else:
                    client_watermark_hook = watermark_hook.with_weight(float(assignment["loss_weight"]))
            state_dict, loss_value = _train_one_client(
                global_model=server.model,
                client=client,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                batch_size=batch_size,
                watermark_hook=client_watermark_hook,
            )
            local_states.append(state_dict)
            local_loss_summaries.append(loss_value)

        mean_task_loss = sum(item["task_loss"] for item in local_loss_summaries) / len(local_loss_summaries)
        mean_wm_loss = sum(item["wm_loss"] for item in local_loss_summaries) / len(local_loss_summaries)
        mean_total_loss = sum(item["total_loss"] for item in local_loss_summaries) / len(local_loss_summaries)

        server.model.load_state_dict(_average_state_dicts(local_states))
        val_accuracy = evaluate_accuracy(
            model=server.model, features=val_features, labels=val_labels
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state": server.model.state_dict(),
                    "model_name": model_name,
                    "num_classes": num_classes,
                    "input_shape": input_shape,
                },
                best_checkpoint_path,
            )
        round_metrics.append(
            {
                "round": round_id,
                "selected_clients": len(selected_clients),
                "mean_client_loss": mean_total_loss,
                "task_loss": mean_task_loss,
                "wm_loss": mean_wm_loss,
                "mean_task_loss": mean_task_loss,
                "mean_wm_loss": mean_wm_loss,
                "val_accuracy": val_accuracy,
                "allocation_enabled": allocation_enabled,
                "allocated_clients": sum(
                    int(item["enabled"]) for item in assignment_for_round.values()
                ),
            }
        )

    torch.save(
        {
            "model_state": server.model.state_dict(),
            "model_name": model_name,
            "num_classes": num_classes,
            "input_shape": input_shape,
        },
        checkpoint_path,
    )
    if watermark_hook is not None:
        save_owner_artifacts(
            path=run_dir / "owner_artifacts.json",
            owner_id=watermark_hook.owner_id,
            codebook=watermark_hook.codebook,
            queries=watermark_hook.positive_queries,
            negative_queries=watermark_hook.negative_queries,
            wm_train_config={"task_weight": 1.0, "wm_weight": watermark_hook.wm_weight},
        )
    write_json(
        metrics_path,
        {
            "dataset": dataset_name,
            "dataset_name": dataset_name,
            "model": model_name,
            "seed": seed,
            "rounds": round_metrics,
        },
    )
    write_json(
        metadata_path,
        {
            "dataset": dataset_name,
            "dataset_name": dataset_name,
            "model": model_name,
            "seed": seed,
            "num_clients": num_clients,
            "rounds": rounds,
            "participation_rate": participation_rate,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "samples_per_client": samples_per_client,
            "num_classes": num_classes,
            "data_source": "dataset-backed",
            "partition": {
                "split_method": partition_metadata.split_method,
                "partition_type": partition_metadata.partition_type,
                "partition_params": partition_metadata.partition_params,
                "seed": partition_metadata.seed,
                "num_clients": partition_metadata.num_clients,
                "total_samples": loaded_dataset.num_samples,
                "selected_samples": len(sampled_indices),
                "client_indices": dataset_partition.client_indices,
                "sampled_dataset_indices": sampled_indices,
                "client_sample_counts": dataset_partition.client_sample_counts,
                "client_label_histograms": dataset_partition.client_label_histograms,
            },
            "allocation": {
                "enabled": allocation_enabled,
                "budget_ratio": allocation_budget_ratio,
                "base_loss_weight": allocation_base_loss_weight,
            },
        },
    )

    if allocation_enabled:
        write_json(
            allocation_path,
            {
                "config": {
                    "enabled": True,
                    "budget_ratio": allocation_budget_ratio,
                    "base_loss_weight": allocation_base_loss_weight,
                },
                "round_assignments": round_assignments,
            },
        )

    return FedAvgResult(
        run_dir=run_dir,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        allocation_path=allocation_path if allocation_enabled else None,
    )
