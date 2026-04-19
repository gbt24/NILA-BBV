import json
from pathlib import Path

from bbv.federated import train_federated


def test_train_federated_exports_validation_metrics_and_best_checkpoint(
    tmp_path: Path,
) -> None:
    result = train_federated(
        output_root=tmp_path / "outputs",
        seed=0,
        dataset_name="cifar10",
        model_name="resnet18",
        num_classes=10,
        num_clients=4,
        rounds=2,
        participation_rate=0.5,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
        samples_per_client=16,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "val_accuracy" in metrics["rounds"][0]
    assert result.best_checkpoint_path.exists()
