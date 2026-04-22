from __future__ import annotations

import torch


def run_quantization_attack(
    *, state_dict: dict[str, torch.Tensor], seed: int, levels: int
) -> tuple[dict[str, torch.Tensor], dict[str, int | str]]:
    del seed

    attacked = {key: value.clone() for key, value in state_dict.items()}
    for key, value in attacked.items():
        if not torch.is_floating_point(value):
            continue
        scale = float(value.abs().max().item())
        if scale == 0.0:
            continue
        step = scale / float(levels)
        attacked[key] = torch.round(value / step) * step
    return attacked, {"attack_name": "quantization", "levels": levels}
