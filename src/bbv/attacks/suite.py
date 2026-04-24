"""Minimal attack suite for Phase 6 robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch

from bbv.attacks.distillation import run_distillation_attack
from bbv.attacks.extraction import run_extraction_attack
from bbv.attacks.finetune import run_finetune_attack
from bbv.attacks.pruning import run_pruning_attack
from bbv.attacks.quantization import run_quantization_attack
from bbv.utils.io import write_json


@dataclass(frozen=True)
class AttackResult:
    output_dir: Path
    attacked_checkpoint: Path
    attack_log: Path


def _resolve_attack_device(device: str) -> torch.device:
    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    raise ValueError("device must be one of: auto, cpu, cuda")


def _attack_state_dict(
    *,
    attack_name: str,
    checkpoint: dict[str, object],
    seed: int,
    dataset_name: str,
    attack_config: dict[str, object] | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[str, torch.Tensor], dict[str, float | int | str]]:
    attack_config = attack_config or {}
    state_dict = checkpoint["model_state"]
    if attack_name == "finetune":
        return run_finetune_attack(
            state_dict=state_dict,
            model_name=str(checkpoint.get("model_name", "mlp")),
            num_classes=int(checkpoint.get("num_classes", 10)),
            input_shape=tuple(checkpoint.get("input_shape", (3, 32, 32))),
            dataset_name=dataset_name,
            seed=seed,
            learning_rate=float(attack_config.get("learning_rate", 0.01)),
            local_epochs=int(attack_config.get("local_epochs", 1)),
            batch_size=int(attack_config.get("batch_size", 8)),
            max_batches=int(attack_config.get("max_batches", 4)),
            device=device,
        )
    if attack_name == "pruning":
        return run_pruning_attack(state_dict=state_dict, seed=seed, ratio=0.2)
    if attack_name == "quantization":
        return run_quantization_attack(state_dict=state_dict, seed=seed, levels=128)
    if attack_name == "distillation":
        return run_distillation_attack(
            state_dict=state_dict,
            model_name=str(checkpoint.get("model_name", "mlp")),
            num_classes=int(checkpoint.get("num_classes", 10)),
            input_shape=tuple(checkpoint.get("input_shape", (3, 32, 32))),
            dataset_name=dataset_name,
            seed=seed,
            student_model_name=str(attack_config.get("student_model_name", checkpoint.get("model_name", "mlp"))),
            learning_rate=float(attack_config.get("learning_rate", 0.01)),
            local_epochs=int(attack_config.get("local_epochs", 1)),
            batch_size=int(attack_config.get("batch_size", 8)),
            max_batches=int(attack_config.get("max_batches", 4)),
            temperature=float(attack_config.get("temperature", 1.0)),
            teacher_query_mode=str(attack_config.get("teacher_query_mode", "logits")),
            device=device,
        )
    if attack_name == "extraction":
        return run_extraction_attack(
            state_dict=state_dict,
            model_name=str(checkpoint.get("model_name", "mlp")),
            num_classes=int(checkpoint.get("num_classes", 10)),
            input_shape=tuple(checkpoint.get("input_shape", (3, 32, 32))),
            dataset_name=dataset_name,
            seed=seed,
            student_model_name=str(attack_config.get("student_model_name", checkpoint.get("model_name", "mlp"))),
            temperature=float(attack_config.get("temperature", 1.0)),
            learning_rate=float(attack_config.get("learning_rate", 0.01)),
            local_epochs=int(attack_config.get("local_epochs", 1)),
            batch_size=int(attack_config.get("batch_size", 8)),
            query_budget=int(attack_config.get("query_budget", 32)),
            teacher_query_mode=str(attack_config.get("teacher_query_mode", "hard-label")),
            device=device,
        )
    raise ValueError(f"unsupported attack: {attack_name}")


def run_attack(
    *,
    attack_name: str,
    checkpoint_path: Path,
    output_root: Path,
    seed: int,
    dataset_name: str = "cifar10",
    attack_config: dict[str, object] | None = None,
    device: str = "auto",
) -> AttackResult:
    attack_device = _resolve_attack_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise ValueError("checkpoint must contain model_state")

    attacked_state, attack_config = _attack_state_dict(
        attack_name=attack_name,
        checkpoint=checkpoint,
        seed=seed,
        dataset_name=dataset_name,
        attack_config=attack_config,
        device=attack_device,
    )

    run_id = f"{attack_name}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    attacked_checkpoint = output_dir / "attacked_checkpoint.pt"
    attack_log = output_dir / "attack_log.json"

    attacked_checkpoint_payload = dict(checkpoint)
    attacked_checkpoint_payload["model_state"] = attacked_state
    if attack_name in {"distillation", "extraction"} and "student_model_name" in attack_config:
        attacked_checkpoint_payload["model_name"] = attack_config["student_model_name"]
    torch.save(attacked_checkpoint_payload, attacked_checkpoint)
    write_json(
        attack_log,
        {
            "attack": attack_name,
            "dataset": dataset_name,
            "attack_config": attack_config,
            "source_checkpoint": str(checkpoint_path),
            "attacked_checkpoint": str(attacked_checkpoint),
            "seed": seed,
        },
    )
    return AttackResult(
        output_dir=output_dir,
        attacked_checkpoint=attacked_checkpoint,
        attack_log=attack_log,
    )
