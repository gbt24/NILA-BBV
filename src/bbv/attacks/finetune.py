from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbv.datasets.loaders import load_dataset
from bbv.models import build_model


def run_finetune_attack(
    *,
    state_dict: dict[str, torch.Tensor],
    model_name: str,
    num_classes: int,
    input_shape: tuple[int, ...],
    dataset_name: str,
    seed: int,
    learning_rate: float,
    local_epochs: int,
    batch_size: int,
    max_batches: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
    torch.manual_seed(seed)
    model = build_model(model_name, num_classes=num_classes, input_shape=input_shape)
    model.load_state_dict(state_dict)
    model.train()

    loaded_dataset = load_dataset(
        root=Path("data/raw"),
        train=True,
        download=True,
        name=dataset_name,
    )
    data_loader = DataLoader(loaded_dataset.dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_optimizer_steps = 0
    last_loss = 0.0
    for _ in range(local_epochs):
        for features, labels in data_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            num_optimizer_steps += 1
            last_loss = float(loss.detach().item())
            if num_optimizer_steps >= max_batches:
                break
        if num_optimizer_steps >= max_batches:
            break

    attacked = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return attacked, {
        "attack_name": "finetune",
        "learning_rate": learning_rate,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "num_optimizer_steps": num_optimizer_steps,
        "last_loss": last_loss,
        "dataset_name": dataset_name,
        "source_split": getattr(loaded_dataset, "split_name", "train"),
    }
