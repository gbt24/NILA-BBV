from __future__ import annotations

import torch


def run_extraction_attack(
    *, state_dict: dict[str, torch.Tensor], seed: int, temperature: float, student_mix: float
) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
    del seed

    attacked = {key: value.clone() for key, value in state_dict.items()}
    for key, value in attacked.items():
        if not torch.is_floating_point(value):
            continue

        flat_teacher_logits = value.reshape(-1).float() / max(temperature, 1e-6)
        teacher_probs = torch.softmax(flat_teacher_logits, dim=0).reshape_as(value)
        scale = max(float(value.abs().max().item()), 1.0)
        student_fit = teacher_probs * scale
        attacked[key] = (student_mix * student_fit) + ((1.0 - student_mix) * value)

    return attacked, {
        "attack_name": "extraction",
        "temperature": temperature,
        "student_mix": student_mix,
    }
