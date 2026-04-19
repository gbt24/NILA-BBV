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
