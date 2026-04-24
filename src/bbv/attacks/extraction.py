from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbv.datasets.loaders import load_dataset
from bbv.models import build_model


def run_extraction_attack(
    *,
    state_dict: dict[str, torch.Tensor],
    model_name: str,
    num_classes: int,
    input_shape: tuple[int, ...],
    dataset_name: str,
    seed: int,
    student_model_name: str,
    temperature: float,
    learning_rate: float,
    local_epochs: int,
    batch_size: int,
    query_budget: int,
    teacher_query_mode: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
    torch.manual_seed(seed)

    teacher = build_model(model_name, num_classes=num_classes, input_shape=input_shape)
    teacher.to(device)
    try:
        teacher.load_state_dict(state_dict)
    except RuntimeError:
        attacked = {key: value.detach().clone() for key, value in state_dict.items()}
        return attacked, {
            "attack_name": "extraction",
            "temperature": temperature,
            "student_model_name": student_model_name,
            "teacher_query_mode": teacher_query_mode,
            "learning_rate": learning_rate,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "query_budget": query_budget,
            "used_queries": 0,
            "num_student_steps": 0,
            "last_loss": 0.0,
            "dataset_name": dataset_name,
            "source_split": "unavailable",
            "fallback_used": "state-dict-incompatible-checkpoint",
        }
    teacher.eval()

    student = build_model(student_model_name, num_classes=num_classes, input_shape=input_shape)
    student.to(device)
    student.train()

    loaded_dataset = load_dataset(
        root=Path("data/raw"),
        train=True,
        download=True,
        name=dataset_name,
    )
    data_loader = DataLoader(loaded_dataset.dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.SGD(student.parameters(), lr=learning_rate)

    query_budget = max(int(query_budget), 1)
    queried_examples = 0
    num_student_steps = 0
    last_loss = 0.0
    temperature = max(float(temperature), 1e-6)

    for _ in range(local_epochs):
        for features, _labels in data_loader:
            remaining = query_budget - queried_examples
            if remaining <= 0:
                break
            features = features[:remaining].to(device)
            with torch.no_grad():
                teacher_logits = teacher(features)

            optimizer.zero_grad()
            student_logits = student(features)
            if teacher_query_mode == "logits":
                teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
                student_log_probs = torch.log_softmax(student_logits / temperature, dim=1)
                loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
            else:
                teacher_targets = teacher_logits.argmax(dim=1)
                loss = F.cross_entropy(student_logits, teacher_targets)
            loss.backward()
            optimizer.step()

            queried_examples += int(features.shape[0])
            num_student_steps += 1
            last_loss = float(loss.detach().item())
            if queried_examples >= query_budget:
                break
        if queried_examples >= query_budget:
            break

    attacked = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    return attacked, {
        "attack_name": "extraction",
        "temperature": temperature,
        "student_model_name": student_model_name,
        "teacher_query_mode": teacher_query_mode,
        "learning_rate": learning_rate,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "query_budget": query_budget,
        "used_queries": queried_examples,
        "num_student_steps": num_student_steps,
        "last_loss": last_loss,
        "dataset_name": dataset_name,
        "source_split": getattr(loaded_dataset, "split_name", "train"),
    }
