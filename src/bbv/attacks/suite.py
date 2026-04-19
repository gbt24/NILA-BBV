"""Minimal attack suite for Phase 6 robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch

from bbv.attacks.distillation import run_distillation_attack
from bbv.attacks.finetune import run_finetune_attack
from bbv.attacks.pruning import run_pruning_attack
from bbv.attacks.quantization import run_quantization_attack
from bbv.utils.io import write_json


@dataclass(frozen=True)
class AttackResult:
    output_dir: Path
    attacked_checkpoint: Path
    attack_log: Path


def _attack_state_dict(
    *, attack_name: str, state_dict: dict[str, torch.Tensor], generator: torch.Generator
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if attack_name == "finetune":
        config = {"noise_scale": 0.01}
        return (
            run_finetune_attack(
                state_dict=state_dict, generator=generator, noise_scale=config["noise_scale"]
            ),
            config,
        )
    if attack_name == "pruning":
        config = {"ratio": 0.2}
        return run_pruning_attack(state_dict=state_dict, ratio=config["ratio"]), config
    if attack_name == "quantization":
        config = {"levels": 128.0}
        return (
            run_quantization_attack(state_dict=state_dict, levels=int(config["levels"])),
            config,
        )
    if attack_name == "distillation":
        config = {"retention": 0.9}
        return (
            run_distillation_attack(state_dict=state_dict, retention=config["retention"]),
            config,
        )
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
    attacked_state, attack_config = _attack_state_dict(
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
            "attack_config": attack_config,
            "source_checkpoint": str(checkpoint_path),
            "attacked_checkpoint": str(attacked_checkpoint),
        },
    )
    return AttackResult(
        output_dir=output_dir,
        attacked_checkpoint=attacked_checkpoint,
        attack_log=attack_log,
    )
