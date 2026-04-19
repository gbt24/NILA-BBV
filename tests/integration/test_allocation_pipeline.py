import json
from pathlib import Path

from bbv.federated import train_federated


def test_allocation_pipeline_writes_assignment_logs_when_enabled(tmp_path: Path) -> None:
    result = train_federated(
        output_root=tmp_path / "outputs",
        seed=0,
        dataset_name="cifar10",
        model_name="resnet18",
        num_classes=10,
        num_clients=5,
        rounds=2,
        participation_rate=0.6,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
        samples_per_client=12,
        allocation_enabled=True,
        allocation_budget_ratio=0.4,
        allocation_base_loss_weight=0.15,
    )

    assert result.allocation_path is not None
    assert result.allocation_path.exists()
    payload = json.loads(result.allocation_path.read_text(encoding="utf-8"))
    assert len(payload["round_assignments"]) == 2
    assert payload["config"]["enabled"] is True


def test_allocation_pipeline_keeps_baseline_when_disabled(tmp_path: Path) -> None:
    result = train_federated(
        output_root=tmp_path / "outputs",
        seed=0,
        dataset_name="cifar10",
        model_name="resnet18",
        num_classes=10,
        num_clients=4,
        rounds=1,
        participation_rate=0.5,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
        samples_per_client=12,
        allocation_enabled=False,
        allocation_budget_ratio=0.5,
        allocation_base_loss_weight=0.2,
    )

    assert result.allocation_path is None
    assert result.checkpoint_path.exists()
