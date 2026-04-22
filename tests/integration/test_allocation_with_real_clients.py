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
    assert "skew_ratio" in first_assignment["stats"]
    assert "main_wm_alignment" in first_assignment["stats"]
    assert "privacy_penalty" in first_assignment["stats"]
    assert -1.0 <= first_assignment["stats"]["main_wm_alignment"] <= 1.0


def test_allocation_enables_one_client_for_small_positive_budget_ratio(tmp_path: Path) -> None:
    result = train_federated(
        output_root=tmp_path / "outputs",
        seed=0,
        dataset_name="cifar10",
        model_name="resnet18",
        num_classes=10,
        num_clients=2,
        rounds=1,
        participation_rate=0.5,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
        samples_per_client=12,
        allocation_enabled=True,
        allocation_budget_ratio=0.5,
        allocation_base_loss_weight=0.15,
    )

    payload = json.loads(result.allocation_path.read_text(encoding="utf-8"))
    assignments = payload["round_assignments"][0]["assignments"]

    assert sum(int(item["enabled"]) for item in assignments.values()) == 1


def test_allocation_logged_stats_match_full_client_partition(tmp_path: Path) -> None:
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
        batch_size=2,
        learning_rate=0.05,
        samples_per_client=12,
        allocation_enabled=True,
        allocation_budget_ratio=0.4,
        allocation_base_loss_weight=0.15,
    )

    payload = json.loads(result.allocation_path.read_text(encoding="utf-8"))
    round_assignment = payload["round_assignments"][0]
    client_id, assignment = next(iter(round_assignment["assignments"].items()))
    stats = assignment["stats"]

    histogram = {
        key: int(value)
        for key, value in round_assignment["selected_histograms_by_client"][client_id].items()
    }
    total = sum(histogram.values())

    assert stats["class_coverage"] == len(histogram)
    assert stats["skew_ratio"] == max(histogram.values()) / total
