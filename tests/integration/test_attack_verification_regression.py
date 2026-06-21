from pathlib import Path
import json
from types import SimpleNamespace

import torch

from bbv.attacks import run_attack
from bbv.federated import train_federated
from bbv.verification import run_verification_from_checkpoint
from bbv.watermarking import build_negative_queries, build_positive_queries, generate_codebook, save_owner_artifacts


class _FakeDataset:
    def __init__(self, size: int) -> None:
        self.classes = [str(index) for index in range(10)]
        self.targets = [index % 10 for index in range(size)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 32, 32), int(self.targets[index])


def _fake_load(*, name: str, root: Path, train: bool, download: bool, train_size: int = 40, test_size: int = 16):
    size = train_size if train else test_size
    return SimpleNamespace(
        dataset_name=name,
        split_name="train" if train else "test",
        train=train,
        num_classes=10,
        num_samples=size,
        dataset=_FakeDataset(size),
    )


def _patch_fake_dataset(monkeypatch, *, train_size: int = 40, test_size: int = 16) -> None:
    import functools
    monkeypatch.setattr(
        "bbv.federated.fedavg.load_dataset",
        functools.partial(_fake_load, train_size=train_size, test_size=test_size),
        raising=False,
    )


def test_attack_outputs_keep_verification_chain_alive(monkeypatch, tmp_path: Path) -> None:
    _patch_fake_dataset(monkeypatch)
    train_result = train_federated(
        output_root=tmp_path / "runs",
        seed=0,
        dataset_name="cifar10",
        model_name="resnet18",
        num_classes=10,
        num_clients=3,
        rounds=1,
        participation_rate=1.0,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
    )
    codebook = generate_codebook(owner_id="owner0", code_length=8, seed=0)
    artifacts = train_result.run_dir / "owner_artifacts.json"
    save_owner_artifacts(
        path=artifacts,
        owner_id="owner0",
        codebook=codebook,
        queries=build_positive_queries(codebook=codebook, seed=0),
        negative_queries=build_negative_queries(codebook=codebook, seed=0),
    )

    attack_result = run_attack(
        attack_name="quantization",
        checkpoint_path=train_result.checkpoint_path,
        output_root=tmp_path / "attacks",
        seed=0,
    )
    summary = run_verification_from_checkpoint(
        checkpoint_path=attack_result.attacked_checkpoint,
        artifacts_path=artifacts,
        verification_path=attack_result.output_dir / "verification_after_attack.json",
        calibration_path=attack_result.output_dir / "calibration_after_attack.json",
        decision_threshold=0.5,
        margin=0.05,
        competitor_owner_ids=["owner1"],
        seed=0,
    )
    attack_log = json.loads(attack_result.attack_log.read_text(encoding="utf-8"))
    assert attack_log["attack_config"]["attack_name"] == "quantization"
    assert "owner_score" in summary
    assert (attack_result.output_dir / "verification_after_attack.json").exists()


def test_attack_outputs_preserve_checkpoint_metadata_for_verification(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_fake_dataset(monkeypatch)
    import bbv.attacks.distillation as distillation_module
    monkeypatch.setattr(distillation_module, "load_dataset", _fake_load, raising=False)
    train_result = train_federated(
        output_root=tmp_path / "runs",
        seed=0,
        dataset_name="cifar10",
        model_name="mlp",
        num_classes=10,
        num_clients=3,
        rounds=1,
        participation_rate=1.0,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
    )
    codebook = generate_codebook(owner_id="owner0", code_length=8, seed=0)
    artifacts = train_result.run_dir / "owner_artifacts.json"
    save_owner_artifacts(
        path=artifacts,
        owner_id="owner0",
        codebook=codebook,
        queries=build_positive_queries(codebook=codebook, seed=0),
        negative_queries=build_negative_queries(codebook=codebook, seed=0),
    )

    attack_result = run_attack(
        attack_name="distillation",
        checkpoint_path=train_result.checkpoint_path,
        output_root=tmp_path / "attacks",
        seed=0,
    )
    summary = run_verification_from_checkpoint(
        checkpoint_path=attack_result.attacked_checkpoint,
        artifacts_path=artifacts,
        verification_path=attack_result.output_dir / "verification_after_attack.json",
        calibration_path=attack_result.output_dir / "calibration_after_attack.json",
        decision_threshold=0.5,
        margin=0.05,
        competitor_owner_ids=["owner1"],
        seed=0,
    )

    assert "owner_score" in summary
