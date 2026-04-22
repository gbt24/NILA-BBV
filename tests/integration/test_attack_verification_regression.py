from pathlib import Path
import json

from bbv.attacks import run_attack
from bbv.federated import train_federated
from bbv.verification import run_verification_from_checkpoint
from bbv.watermarking import build_negative_queries, build_positive_queries, generate_codebook, save_owner_artifacts


def test_attack_outputs_keep_verification_chain_alive(tmp_path: Path) -> None:
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
        samples_per_client=12,
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


def test_attack_outputs_preserve_checkpoint_metadata_for_verification(tmp_path: Path) -> None:
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
        samples_per_client=12,
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
