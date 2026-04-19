from __future__ import annotations

import torch


def run_quantization_attack(
    *, state_dict: dict[str, torch.Tensor], levels: int
) -> dict[str, torch.Tensor]:
    attacked = {key: value.clone() for key, value in state_dict.items()}
    for key, value in attacked.items():
        if not torch.is_floating_point(value):
            continue
        scale = float(value.abs().max().item())
        if scale == 0.0:
            continue
        step = scale / float(levels)
        attacked[key] = torch.round(value / step) * step
    return attacked
