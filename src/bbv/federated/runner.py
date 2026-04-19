"""Minimal federated smoke runner for Phase 0."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from bbv.models.simple import build_simple_classifier
from bbv.utils.io import RunPaths, create_run_paths, write_json


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    summary_path: Path
    metrics: dict[str, float | int]


def _client_batch(
    client_index: int, samples_per_client: int, input_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.randn(samples_per_client, input_dim)
    labels = ((features.sum(dim=1) + client_index) > 0).long() % 2
    return features, labels


def _train_client(
    global_model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
    local_epochs: int,
) -> tuple[dict[str, torch.Tensor], float]:
    client_model = deepcopy(global_model)
    optimizer = torch.optim.SGD(client_model.parameters(), lr=learning_rate)
    last_loss = 0.0

    for _ in range(local_epochs):
        optimizer.zero_grad()
        logits = client_model(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())

    return client_model.state_dict(), last_loss


def _average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        averaged[key] = torch.stack(
            [state_dict[key] for state_dict in state_dicts], dim=0
        ).mean(dim=0)
    return averaged


def _build_summary(
    paths: RunPaths, metrics: dict[str, float | int], seed: int
) -> RunResult:
    payload = {
        "seed": seed,
        "run_dir": str(paths.run_dir),
        "metrics": metrics,
    }
    write_json(paths.summary_path, payload)
    return RunResult(
        run_dir=paths.run_dir, summary_path=paths.summary_path, metrics=metrics
    )


def _validate_smoke_inputs(
    num_clients: int,
    samples_per_client: int,
    local_epochs: int,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    learning_rate: float,
) -> None:
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0")
    if samples_per_client <= 0:
        raise ValueError("samples_per_client must be greater than 0")
    if local_epochs <= 0:
        raise ValueError("local_epochs must be greater than 0")
    if input_dim <= 0:
        raise ValueError("input_dim must be greater than 0")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be greater than 0")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")


def run_smoke_experiment(
    output_root: Path,
    seed: int = 7,
    num_clients: int = 3,
    samples_per_client: int = 8,
    input_dim: int = 4,
    hidden_dim: int = 16,
    num_classes: int = 2,
    learning_rate: float = 0.1,
    local_epochs: int = 1,
) -> RunResult:
    _validate_smoke_inputs(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        local_epochs=local_epochs,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )
    torch.manual_seed(seed)
    output_root = Path(output_root)
    paths = create_run_paths(output_root)

    global_model = build_simple_classifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )

    client_states: list[dict[str, torch.Tensor]] = []
    client_losses: list[float] = []
    for client_index in range(num_clients):
        features, labels = _client_batch(client_index, samples_per_client, input_dim)
        state_dict, loss_value = _train_client(
            global_model=global_model,
            features=features,
            labels=labels,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
        )
        client_states.append(state_dict)
        client_losses.append(loss_value)

    global_model.load_state_dict(_average_state_dicts(client_states))
    metrics: dict[str, float | int] = {
        "num_clients": num_clients,
        "samples_per_client": samples_per_client,
        "final_loss": sum(client_losses) / len(client_losses),
    }
    return _build_summary(paths=paths, metrics=metrics, seed=seed)


def run_smoke_from_config(cfg: DictConfig) -> RunResult:
    return run_smoke_experiment(
        output_root=Path(cfg.output_root),
        seed=int(cfg.seed),
        num_clients=int(cfg.federated.num_clients),
        samples_per_client=int(cfg.federated.samples_per_client),
        input_dim=int(cfg.model.input_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_classes=int(cfg.model.num_classes),
        learning_rate=float(cfg.federated.learning_rate),
        local_epochs=int(cfg.federated.local_epochs),
    )
