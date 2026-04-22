from pathlib import Path
import json

import torch

from bbv.attacks import run_attack


def test_pruning_attack_zeroes_fraction_of_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model_state": {"w": torch.tensor([0.1, 0.2, 0.9, 1.2])}}, checkpoint)

    result = run_attack(
        attack_name="pruning",
        checkpoint_path=checkpoint,
        output_root=tmp_path / "attacks",
        seed=0,
    )
    attacked = torch.load(result.attacked_checkpoint, map_location="cpu")["model_state"]["w"]
    assert (attacked == 0).sum().item() >= 1
    attack_log = json.loads(result.attack_log.read_text(encoding="utf-8"))
    assert "attack_config" in attack_log


def test_run_attack_persists_attack_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model_state": {"w": torch.tensor([0.1, 0.2, 0.9, 1.2])}}, checkpoint)

    result = run_attack(
        attack_name="pruning",
        checkpoint_path=checkpoint,
        output_root=tmp_path / "attacks",
        seed=0,
    )

    payload = json.loads(result.attack_log.read_text())
    assert payload["attack_config"]["attack_name"] == "pruning"


def test_run_attack_log_contains_minimal_protocol_fields(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model_state": {"w": torch.tensor([0.1, 0.2, 0.9, 1.2])}}, checkpoint)

    result = run_attack(
        attack_name="extraction",
        checkpoint_path=checkpoint,
        output_root=tmp_path / "attacks",
        seed=3,
    )

    payload = json.loads(result.attack_log.read_text())
    assert payload["attack"] == "extraction"
    assert payload["source_checkpoint"] == str(checkpoint)
    assert payload["attacked_checkpoint"] == str(result.attacked_checkpoint)
    assert payload["seed"] == 3
