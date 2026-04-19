from pathlib import Path

from bbv.federated import train_federated
from bbv.verification import run_verification_from_checkpoint
from bbv.watermarking import (
    build_negative_queries,
    build_positive_queries,
    generate_codebook,
    save_owner_artifacts,
)


def test_verification_pipeline_exports_summary_and_calibration(tmp_path: Path) -> None:
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
    pos_queries = build_positive_queries(codebook=codebook, seed=0)
    neg_queries = build_negative_queries(codebook=codebook, seed=0)
    artifacts_path = train_result.run_dir / "owner_artifacts.json"
    save_owner_artifacts(
        path=artifacts_path,
        owner_id="owner0",
        codebook=codebook,
        queries=pos_queries,
        negative_queries=neg_queries,
    )

    result = run_verification_from_checkpoint(
        checkpoint_path=train_result.checkpoint_path,
        artifacts_path=artifacts_path,
        verification_path=train_result.run_dir / "verification_margin_summary.json",
        calibration_path=train_result.run_dir / "calibration_artifacts.json",
        decision_threshold=0.5,
        margin=0.05,
        competitor_owner_ids=["owner1", "owner2"],
        seed=0,
    )

    assert "owner_score" in result
    assert "competitor_scores" in result
    assert (train_result.run_dir / "verification_margin_summary.json").exists()
    assert (train_result.run_dir / "calibration_artifacts.json").exists()
