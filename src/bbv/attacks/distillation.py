from __future__ import annotations

import torch


def run_distillation_attack(
    *, state_dict: dict[str, torch.Tensor], seed: int, retention: float
) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
    del seed

    attacked = {key: value.clone() for key, value in state_dict.items()}
    for key, value in attacked.items():
        if torch.is_floating_point(value):
            attacked[key] = value * retention
    return attacked, {"attack_name": "distillation", "retention": retention}
