import json
from pathlib import Path

from bbv.federated import train_federated


def test_allocation_assignments_include_stats_fields(tmp_path: Path) -> None:
    result = train_federated(
        output_root=tmp_path / "outputs",
        seed=0,
        dataset_name="cifar10",
        model_name="resnet18",
        num_classes=10,
        num_clients=5,
        rounds=1,
        participation_rate=0.6,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
        samples_per_client=12,
        allocation_enabled=True,
        allocation_budget_ratio=0.4,
        allocation_base_loss_weight=0.15,
    )

    payload = json.loads(result.allocation_path.read_text(encoding="utf-8"))
    assignments = payload["round_assignments"][0]["assignments"]
    first_assignment = next(iter(assignments.values()))
    assert "stats" in first_assignment
    assert "class_coverage" in first_assignment["stats"]
