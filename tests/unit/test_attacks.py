from pathlib import Path

import torch

from bbv.attacks import run_attack


def test_run_attack_writes_attacked_checkpoint_and_log(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {"linear.weight": torch.ones(2, 2)}}, checkpoint_path)

    result = run_attack(
        attack_name="finetune",
        checkpoint_path=checkpoint_path,
        output_root=tmp_path / "attack_outputs",
        seed=0,
    )

    assert result.attacked_checkpoint.exists()
    assert result.attack_log.exists()
    assert checkpoint_path.exists()


def test_run_attack_supports_required_attack_types(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {"linear.weight": torch.randn(2, 2)}}, checkpoint_path)

    for attack_name in ["finetune", "pruning", "quantization", "distillation", "extraction"]:
        result = run_attack(
            attack_name=attack_name,
            checkpoint_path=checkpoint_path,
            output_root=tmp_path / "attack_outputs",
            seed=7,
        )
        assert result.attacked_checkpoint.exists()
