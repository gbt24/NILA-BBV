"""Minimal attack suite for Phase 6 robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch

from bbv.utils.io import write_json


@dataclass(frozen=True)
class AttackResult:
    output_dir: Path
    attacked_checkpoint: Path
    attack_log: Path


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in state_dict.items()}


def _apply_finetune_noise(
    state_dict: dict[str, torch.Tensor], generator: torch.Generator
) -> dict[str, torch.Tensor]:
    attacked = _clone_state_dict(state_dict)
    for key, value in attacked.items():
        if torch.is_floating_point(value):
            attacked[key] = value + 0.01 * torch.randn_like(value, generator=generator)
    return attacked


def _apply_pruning(state_dict: dict[str, torch.Tensor], ratio: float = 0.2) -> dict[str, torch.Tensor]:
    attacked = _clone_state_dict(state_dict)
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


def _apply_quantization(state_dict: dict[str, torch.Tensor], levels: int = 128) -> dict[str, torch.Tensor]:
    attacked = _clone_state_dict(state_dict)
    for key, value in attacked.items():
        if not torch.is_floating_point(value):
            continue
        scale = float(value.abs().max().item())
        if scale == 0.0:
            continue
        step = scale / float(levels)
        attacked[key] = torch.round(value / step) * step
    return attacked


def _apply_distillation(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    attacked = _clone_state_dict(state_dict)
    for key, value in attacked.items():
        if torch.is_floating_point(value):
            attacked[key] = value * 0.9
    return attacked


def _attack_state_dict(
    *, attack_name: str, state_dict: dict[str, torch.Tensor], generator: torch.Generator
) -> dict[str, torch.Tensor]:
    if attack_name == "finetune":
        return _apply_finetune_noise(state_dict=state_dict, generator=generator)
    if attack_name == "pruning":
        return _apply_pruning(state_dict=state_dict)
    if attack_name == "quantization":
        return _apply_quantization(state_dict=state_dict)
    if attack_name == "distillation":
        return _apply_distillation(state_dict=state_dict)
    raise ValueError(f"unsupported attack: {attack_name}")


def run_attack(
    *,
    attack_name: str,
    checkpoint_path: Path,
    output_root: Path,
    seed: int,
    dataset_name: str = "cifar10",
) -> AttackResult:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise ValueError("checkpoint must contain model_state")

    generator = torch.Generator().manual_seed(seed)
    attacked_state = _attack_state_dict(
        attack_name=attack_name,
        state_dict=checkpoint["model_state"],
        generator=generator,
    )

    run_id = f"{attack_name}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    attacked_checkpoint = output_dir / "attacked_checkpoint.pt"
    attack_log = output_dir / "attack_log.json"

    torch.save({"model_state": attacked_state}, attacked_checkpoint)
    write_json(
        attack_log,
        {
            "attack": attack_name,
            "dataset": dataset_name,
            "seed": seed,
            "source_checkpoint": str(checkpoint_path),
            "attacked_checkpoint": str(attacked_checkpoint),
        },
    )
    return AttackResult(
        output_dir=output_dir,
        attacked_checkpoint=attacked_checkpoint,
        attack_log=attack_log,
    )
