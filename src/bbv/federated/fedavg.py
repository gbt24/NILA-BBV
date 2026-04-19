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

from bbv.allocation import allocate_watermark_budget, estimate_adaptability
from bbv.models import build_model
from bbv.utils.io import write_json


@dataclass(frozen=True)
class FederatedClient:
    client_id: int
    features: torch.Tensor
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
    allocation_path: Path | None


def build_model_input_shape(dataset_name: str) -> tuple[int, int, int]:
    if dataset_name.lower() != "cifar10":
        raise ValueError(f"unsupported dataset: {dataset_name}")
    return (3, 32, 32)


def build_client(
    *,
    client_id: int,
    dataset_name: str,
    samples_per_client: int,
    num_classes: int,
    seed: int,
) -> FederatedClient:
    channels, height, width = build_model_input_shape(dataset_name)
    generator = torch.Generator().manual_seed(seed + client_id)
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
    return FederatedClient(client_id=client_id, features=features, labels=labels)


def build_server(*, model_name: str, num_classes: int, seed: int) -> FederatedServer:
    torch.manual_seed(seed)
    return FederatedServer(model=build_model(model_name, num_classes=num_classes))


def _train_one_client(
    *,
    global_model: torch.nn.Module,
    client: FederatedClient,
    learning_rate: float,
    local_epochs: int,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], float]:
    local_model = deepcopy(global_model)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    data_loader = DataLoader(
        TensorDataset(client.features, client.labels),
        batch_size=batch_size,
        shuffle=True,
    )

    running_losses: list[float] = []
    for _ in range(local_epochs):
        for features, labels in data_loader:
            optimizer.zero_grad()
            logits = local_model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_losses.append(float(loss.detach().item()))
    avg_loss = sum(running_losses) / len(running_losses)
    return local_model.state_dict(), avg_loss


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
    if allocation_budget_ratio <= 0.0 or allocation_budget_ratio > 1.0:
        raise ValueError("allocation_budget_ratio must be in (0, 1]")
    if allocation_base_loss_weight < 0.0:
        raise ValueError("allocation_base_loss_weight must be non-negative")


def _build_label_histogram(labels: torch.Tensor) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for label in labels.tolist():
        key = str(int(label))
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


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
    allocation_enabled: bool = False,
    allocation_budget_ratio: float = 0.3,
    allocation_base_loss_weight: float = 0.1,
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
    allocation_path = run_dir / "allocation_assignments.json"

    clients = [
        build_client(
            client_id=client_id,
            dataset_name=dataset_name,
            samples_per_client=samples_per_client,
            num_classes=num_classes,
            seed=seed,
        )
        for client_id in range(num_clients)
    ]
    server = build_server(model_name=model_name, num_classes=num_classes, seed=seed)

    selected_per_round = max(1, int(round(num_clients * participation_rate)))
    round_metrics: list[dict[str, float | int]] = []
    round_assignments: list[dict[str, object]] = []
    rng = torch.Generator().manual_seed(seed)

    for round_id in range(1, rounds + 1):
        permutation = torch.randperm(num_clients, generator=rng).tolist()
        selected_ids = permutation[:selected_per_round]
        selected_clients = [clients[client_id] for client_id in selected_ids]

        assignment_for_round: dict[int, dict[str, float | int | bool]] = {}
        if allocation_enabled:
            selected_histograms = [
                _build_label_histogram(client.labels) for client in selected_clients
            ]
            adaptability_scores = estimate_adaptability(
                client_label_histograms=selected_histograms
            )
            allocation_budget_clients = max(
                1, int(round(len(selected_clients) * allocation_budget_ratio))
            )
            assignment_local_ids = allocate_watermark_budget(
                adaptability_scores=adaptability_scores,
                budget_clients=allocation_budget_clients,
                base_loss_weight=allocation_base_loss_weight,
            )
            assignment_for_round = {
                selected_ids[local_id]: assignment
                for local_id, assignment in assignment_local_ids.items()
            }
            round_assignments.append(
                {
                    "round": round_id,
                    "selected_client_ids": selected_ids,
                    "assignments": assignment_for_round,
                }
            )

        local_states: list[dict[str, torch.Tensor]] = []
        local_losses: list[float] = []
        for client in selected_clients:
            state_dict, loss_value = _train_one_client(
                global_model=server.model,
                client=client,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                batch_size=batch_size,
            )
            local_states.append(state_dict)
            local_losses.append(loss_value)

        server.model.load_state_dict(_average_state_dicts(local_states))
        round_metrics.append(
            {
                "round": round_id,
                "selected_clients": len(selected_clients),
                "mean_client_loss": sum(local_losses) / len(local_losses),
                "allocation_enabled": allocation_enabled,
                "allocated_clients": sum(
                    int(item["enabled"]) for item in assignment_for_round.values()
                ),
            }
        )

    torch.save({"model_state": server.model.state_dict()}, checkpoint_path)
    write_json(
        metrics_path,
        {
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "rounds": round_metrics,
        },
    )
    write_json(
        metadata_path,
        {
            "dataset": dataset_name,
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
        allocation_path=allocation_path if allocation_enabled else None,
    )
