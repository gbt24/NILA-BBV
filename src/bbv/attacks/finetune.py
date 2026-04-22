from __future__ import annotations

import torch


def run_finetune_attack(
    *, state_dict: dict[str, torch.Tensor], seed: int, noise_scale: float
) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
    generator = torch.Generator().manual_seed(seed)
    attacked = {key: value.clone() for key, value in state_dict.items()}
    for key, value in attacked.items():
        if torch.is_floating_point(value):
            attacked[key] = value + noise_scale * torch.randn_like(value, generator=generator)
    return attacked, {"attack_name": "finetune", "noise_scale": noise_scale}
