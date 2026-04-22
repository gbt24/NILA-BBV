import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bbv.federated import train_federated


def test_train_federated_exports_validation_metrics_and_best_checkpoint(
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
        size = 80 if train else 16
        return SimpleNamespace(
            dataset_name=name,
            split_name="train" if train else "test",
            train=train,
            num_classes=10,
            num_samples=size,
            dataset=FakeDataset(size),
        )

    monkeypatch.setattr("bbv.federated.fedavg.load_dataset", fake_load_dataset, raising=False)

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


def test_train_federated_records_dataset_backed_data_source(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.classes = [str(index) for index in range(10)]
            self.targets = [index % 10 for index in range(40)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        return SimpleNamespace(
            dataset_name=name,
            split_name="train",
            train=train,
            num_classes=10,
            num_samples=40,
            dataset=FakeDataset(),
        )

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
        samples_per_client=10,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["data_source"] == "dataset-backed"


def test_build_client_supports_client_dataset_shape() -> None:
    from bbv.federated.fedavg import build_client

    client = build_client(
        client_id=0,
        dataset_name="cifar10",
        samples_per_client=4,
        num_classes=2,
        seed=0,
    )

    assert client.dataset.client_id == 0
    assert client.dataset.dataset is not None
    assert sum(client.dataset.label_histogram.values()) == 4


def test_train_federated_records_requested_client_sample_counts(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.classes = [str(index) for index in range(10)]
            self.targets = [index % 10 for index in range(40)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        return SimpleNamespace(
            dataset_name=name,
            split_name="train",
            train=train,
            num_classes=10,
            num_samples=40,
            dataset=FakeDataset(),
        )

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
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["partition"]["client_sample_counts"] == [8, 8, 8, 8]


def test_train_federated_rejects_underprovisioned_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.classes = ["0", "1"]
            self.targets = [0, 1, 0, 1, 0, 1]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        return SimpleNamespace(
            dataset_name=name,
            split_name="train",
            train=train,
            num_classes=2,
            num_samples=6,
            dataset=FakeDataset(),
        )

    monkeypatch.setattr("bbv.federated.fedavg.load_dataset", fake_load_dataset, raising=False)

    with pytest.raises(ValueError, match="samples_per_client"):
        train_federated(
            output_root=tmp_path / "outputs",
            seed=0,
            dataset_name="cifar10",
            model_name="resnet18",
            num_classes=2,
            num_clients=4,
            rounds=1,
            participation_rate=0.5,
            local_epochs=1,
            batch_size=2,
            learning_rate=0.05,
            samples_per_client=2,
        )


def test_train_federated_loads_real_validation_split_and_records_sampled_indices(
    monkeypatch, tmp_path: Path
) -> None:
    seen_train_flags: list[bool] = []

    class FakeDataset:
        def __init__(self, size: int) -> None:
            self.classes = [str(index) for index in range(10)]
            self.targets = [index % 10 for index in range(size)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        seen_train_flags.append(train)
        size = 40 if train else 16
        return SimpleNamespace(
            dataset_name=name,
            split_name="train" if train else "test",
            train=train,
            num_classes=10,
            num_samples=size,
            dataset=FakeDataset(size),
        )

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
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    sampled_indices = metadata["partition"]["sampled_dataset_indices"]
    flattened_client_indices = [
        index
        for client_indices in metadata["partition"]["client_indices"]
        for index in client_indices
    ]

    assert seen_train_flags == [True, False]
    assert max(sampled_indices) >= 32
    assert set(flattened_client_indices).issubset(set(sampled_indices))


def test_train_federated_preserves_quantity_skew_client_counts(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.classes = [str(index) for index in range(10)]
            self.targets = [index % 10 for index in range(80)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        size = 80 if train else 16
        return SimpleNamespace(
            dataset_name=name,
            split_name="train" if train else "test",
            train=train,
            num_classes=10,
            num_samples=size,
            dataset=FakeDataset(),
        )

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
        partition_type="quantity_skew",
        quantity_sigma=1.0,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["partition"]["partition_type"] == "quantity_skew"
    assert max(metadata["partition"]["client_sample_counts"]) != min(
        metadata["partition"]["client_sample_counts"]
    )


def test_train_federated_quantity_skew_avoids_singleton_clients_when_budget_allows(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.classes = [str(index) for index in range(10)]
            self.targets = [index % 10 for index in range(80)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        size = 80 if train else 16
        return SimpleNamespace(
            dataset_name=name,
            split_name="train" if train else "test",
            train=train,
            num_classes=10,
            num_samples=size,
            dataset=FakeDataset(),
        )

    monkeypatch.setattr("bbv.federated.fedavg.load_dataset", fake_load_dataset, raising=False)

    result = train_federated(
        output_root=tmp_path / "outputs",
        seed=2,
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
        partition_type="quantity_skew",
        quantity_sigma=1.0,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert min(metadata["partition"]["client_sample_counts"]) >= 2


def test_train_federated_natural_partition_uses_full_dataset_without_sampling(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeDataset:
        def __init__(self, size: int) -> None:
            self.classes = [str(index) for index in range(62)]
            self.targets = [index % 62 for index in range(size)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, name: str, root: Path, train: bool, download: bool):
        size = 48 if train else 16
        return SimpleNamespace(
            dataset_name=name,
            split_name="train" if train else "test",
            train=train,
            num_classes=62,
            num_samples=size,
            dataset=FakeDataset(size),
        )

    monkeypatch.setattr("bbv.federated.fedavg.load_dataset", fake_load_dataset, raising=False)

    result = train_federated(
        output_root=tmp_path / "outputs",
        seed=0,
        dataset_name="femnist",
        model_name="resnet18",
        num_classes=62,
        num_clients=4,
        rounds=1,
        participation_rate=0.5,
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
        samples_per_client=8,
        partition_type="natural",
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["partition"]["partition_type"] == "natural"
    assert metadata["partition"]["selected_samples"] == 48
