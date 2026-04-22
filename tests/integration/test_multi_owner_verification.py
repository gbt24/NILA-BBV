from pathlib import Path

from bbv.federated import train_federated
from bbv.verification import run_verification_from_checkpoint
from bbv.watermarking import (
    build_negative_queries,
    build_positive_queries,
    generate_codebook,
    save_owner_artifacts,
)


def test_multi_owner_verification_exports_margin_and_ambiguity(tmp_path: Path) -> None:
    train_result = train_federated(
        output_root=tmp_path / "outputs",
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
    pos = build_positive_queries(codebook=codebook, seed=0)
    neg = build_negative_queries(codebook=codebook, seed=0)
    artifacts_path = train_result.run_dir / "owner_artifacts.json"
    save_owner_artifacts(
        path=artifacts_path,
        owner_id="owner0",
        codebook=codebook,
        queries=pos,
        negative_queries=neg,
    )

    summary = run_verification_from_checkpoint(
        checkpoint_path=train_result.checkpoint_path,
        artifacts_path=artifacts_path,
        verification_path=train_result.run_dir / "verification_multi_owner.json",
        calibration_path=train_result.run_dir / "calibration_multi_owner.json",
        decision_threshold=0.5,
        margin=0.05,
        competitor_owner_ids=["owner1", "owner2"],
        seed=0,
    )

    assert "margin_value" in summary
    assert "competitor_scores" in summary
    assert "ambiguity_flag" in summary
    assert set(summary["competitor_scores"]) == {"owner1", "owner2"}
    assert "owner_score" in summary


def test_negative_evidence_changes_multi_owner_margin(tmp_path: Path) -> None:
    train_result = train_federated(
        output_root=tmp_path / "outputs",
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
    pos = build_positive_queries(codebook=codebook, seed=0)
    neg = build_negative_queries(codebook=codebook, seed=0)
    artifacts_with_neg = train_result.run_dir / "owner_with_neg.json"
    artifacts_without_neg = train_result.run_dir / "owner_without_neg.json"
    save_owner_artifacts(
        path=artifacts_with_neg,
        owner_id="owner0",
        codebook=codebook,
        queries=pos,
        negative_queries=neg,
    )
    save_owner_artifacts(
        path=artifacts_without_neg,
        owner_id="owner0",
        codebook=codebook,
        queries=pos,
        negative_queries=[],
    )

    with_neg = run_verification_from_checkpoint(
        checkpoint_path=train_result.checkpoint_path,
        artifacts_path=artifacts_with_neg,
        verification_path=train_result.run_dir / "verification_with_neg.json",
        calibration_path=train_result.run_dir / "calibration_with_neg.json",
        decision_threshold=0.5,
        margin=0.05,
        competitor_owner_ids=["owner1", "owner2"],
        seed=0,
    )
    without_neg = run_verification_from_checkpoint(
        checkpoint_path=train_result.checkpoint_path,
        artifacts_path=artifacts_without_neg,
        verification_path=train_result.run_dir / "verification_without_neg.json",
        calibration_path=train_result.run_dir / "calibration_without_neg.json",
        decision_threshold=0.5,
        margin=0.05,
        competitor_owner_ids=["owner1", "owner2"],
        seed=0,
    )

    assert with_neg["margin_value"] != without_neg["margin_value"]
