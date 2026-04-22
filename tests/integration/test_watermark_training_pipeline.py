import json
from pathlib import Path

import torch

from bbv.federated.fedavg import _train_one_client, build_client, build_server
from bbv.federated import train_federated
from bbv.federated.hooks import WatermarkHook


def test_train_one_client_accepts_watermark_hook() -> None:
    hook = WatermarkHook(owner_id="owner0", code_length=8, wm_weight=0.2, seed=0)
    client = build_client(
        client_id=0,
        dataset_name="cifar10",
        samples_per_client=8,
        num_classes=10,
        seed=0,
    )
    server = build_server(model_name="resnet18", num_classes=10, seed=0)

    state_dict, losses = _train_one_client(
        global_model=server.model,
        client=client,
        learning_rate=0.05,
        local_epochs=1,
        batch_size=4,
        watermark_hook=hook,
    )

    assert state_dict
    assert "task_loss" in losses
    assert "wm_loss" in losses


def test_train_federated_records_watermark_metrics_and_owner_artifacts(
    monkeypatch, tmp_path: Path,
) -> None:
    class FakeDataset:
        def __init__(self, size: int) -> None:
            self.classes = [str(index) for index in range(10)]
            self.targets = [index % 10 for index in range(size)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        size = 40 if train else 16
        return type("Loaded", (), {
            "dataset_name": name,
            "split_name": "train" if train else "test",
            "train": train,
            "num_classes": 10,
            "num_samples": size,
            "dataset": FakeDataset(size),
        })()

    monkeypatch.setattr("bbv.federated.fedavg.load_dataset", fake_load_dataset, raising=False)

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
        samples_per_client=8,
        watermark_hook=WatermarkHook(
            owner_id="owner0",
            code_length=8,
            wm_weight=0.2,
            seed=0,
        ),
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    artifacts = json.loads((result.run_dir / "owner_artifacts.json").read_text(encoding="utf-8"))

    assert "task_loss" in metrics["rounds"][0]
    assert "wm_loss" in metrics["rounds"][0]
    assert artifacts["codebook_hash"]
    assert len(artifacts["positive_queries"]) == 8
    assert len(artifacts["negative_queries"]) == 8
    assert artifacts["wm_train_config"]["wm_weight"] == 0.2


def test_train_federated_allocation_can_disable_watermark_loss(
    monkeypatch, tmp_path: Path,
) -> None:
    class FakeDataset:
        def __init__(self, size: int) -> None:
            self.classes = [str(index) for index in range(10)]
            self.targets = [index % 10 for index in range(size)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        size = 40 if train else 16
        return type("Loaded", (), {
            "dataset_name": name,
            "split_name": "train" if train else "test",
            "train": train,
            "num_classes": 10,
            "num_samples": size,
            "dataset": FakeDataset(size),
        })()

    monkeypatch.setattr("bbv.federated.fedavg.load_dataset", fake_load_dataset, raising=False)

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
        samples_per_client=8,
        allocation_enabled=True,
        allocation_budget_ratio=0.0,
        watermark_hook=WatermarkHook(
            owner_id="owner0",
            code_length=8,
            wm_weight=0.2,
            seed=0,
        ),
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert metrics["rounds"][0]["wm_loss"] == 0.0


def test_train_federated_allocation_applies_client_specific_watermark_hooks(
    monkeypatch, tmp_path: Path,
) -> None:
    class FakeDataset:
        def __init__(self, size: int) -> None:
            self.classes = [str(index) for index in range(10)]
            self.targets = [index % 10 for index in range(size)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        size = 40 if train else 16
        return type("Loaded", (), {
            "dataset_name": name,
            "split_name": "train" if train else "test",
            "train": train,
            "num_classes": 10,
            "num_samples": size,
            "dataset": FakeDataset(size),
        })()

    recorded_weights: list[float | None] = []

    def fake_train_one_client(**kwargs):
        hook = kwargs["watermark_hook"]
        recorded_weights.append(None if hook is None else hook.wm_weight)
        return kwargs["global_model"].state_dict(), {
            "task_loss": 1.0,
            "wm_loss": 0.0 if hook is None else hook.wm_weight,
            "total_loss": 1.0 if hook is None else 1.0 + hook.wm_weight,
        }

    monkeypatch.setattr("bbv.federated.fedavg.load_dataset", fake_load_dataset, raising=False)
    monkeypatch.setattr("bbv.federated.fedavg._train_one_client", fake_train_one_client)

    train_federated(
        output_root=tmp_path / "outputs",
        seed=0,
        dataset_name="cifar10",
        model_name="resnet18",
        num_classes=10,
        num_clients=4,
        rounds=1,
        participation_rate=1.0,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
        samples_per_client=8,
        allocation_enabled=True,
        allocation_budget_ratio=0.5,
        allocation_base_loss_weight=0.3,
        watermark_hook=WatermarkHook(
            owner_id="owner0",
            code_length=8,
            wm_weight=0.2,
            seed=0,
        ),
    )

    assert any(weight is None for weight in recorded_weights)
    assert any(weight is not None and weight != 0.2 for weight in recorded_weights)
