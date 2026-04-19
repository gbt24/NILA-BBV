from __future__ import annotations

import torch


def run_pruning_attack(
    *, state_dict: dict[str, torch.Tensor], ratio: float
) -> dict[str, torch.Tensor]:
    attacked = {key: value.clone() for key, value in state_dict.items()}
    for key, value in attacked.items():
        if not torch.is_floating_point(value):
            continue
        flat = value.abs().reshape(-1)
        if flat.numel() == 0:
            continue
        threshold_index = min(int(flat.numel() * ratio), flat.numel() - 1)
        threshold = torch.kthvalue(flat, threshold_index + 1).values.item()
        attacked[key] = torch.where(value.abs() <= threshold, torch.zeros_like(value), value)
    return attacked
