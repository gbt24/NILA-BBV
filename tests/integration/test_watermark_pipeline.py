import hashlib
import json
from pathlib import Path

from bbv.federated import train_federated
from bbv.verification.baseline import verify_owner_from_checkpoint
from bbv.watermarking.baseline import (
    build_positive_queries,
    generate_codebook,
    save_owner_artifacts,
)


def test_watermark_artifacts_and_verification_pipeline(tmp_path: Path) -> None:
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
    queries = build_positive_queries(codebook=codebook, seed=0)
    artifacts_path = train_result.run_dir / "owner_artifacts.json"
    save_owner_artifacts(
        path=artifacts_path,
        owner_id="owner0",
        codebook=codebook,
        queries=queries,
        wm_train_config={"task_weight": 1.0, "wm_weight": 0.2},
    )

    summary = verify_owner_from_checkpoint(
        checkpoint_path=train_result.checkpoint_path,
        artifacts_path=artifacts_path,
        verification_path=train_result.run_dir / "verification_summary.json",
        decision_threshold=0.0,
    )

    assert artifacts_path.exists()
    assert summary["owner_id"] == "owner0"
    assert "score" in summary
    assert (train_result.run_dir / "verification_summary.json").exists()

    payload = json.loads((train_result.run_dir / "verification_summary.json").read_text())
    assert payload["owner_id"] == "owner0"

    artifacts_payload = json.loads(artifacts_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256("".join(str(bit) for bit in codebook).encode("utf-8")).hexdigest()
    assert artifacts_payload["codebook_hash"] == expected_hash
    assert artifacts_payload["wm_train_config"]["wm_weight"] == 0.2
