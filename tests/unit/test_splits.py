from pathlib import Path
from types import SimpleNamespace

import pytest

from bbv.datasets.metadata import (
    SplitMetadata,
    load_split_metadata,
    save_split_metadata,
)


def test_dirichlet_partition_is_reproducible() -> None:
    from bbv.datasets.partitions import build_partition

    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    left = build_partition(labels, num_clients=3, concentration=0.5, seed=7)
    right = build_partition(labels, num_clients=3, concentration=0.5, seed=7)

    assert left.client_indices == right.client_indices
    assert left.client_label_histograms == right.client_label_histograms


def test_dirichlet_partition_conserves_samples() -> None:
    from bbv.datasets.partitions import build_partition

    labels = [0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2]

    result = build_partition(labels, num_clients=4, concentration=0.3, seed=11)
    flattened = sorted(index for indices in result.client_indices for index in indices)

    assert flattened == list(range(len(labels)))
    assert sum(result.client_sample_counts) == len(labels)


def test_dirichlet_partition_changes_with_seed() -> None:
    from bbv.datasets.partitions import build_partition

    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    left = build_partition(labels, num_clients=3, concentration=0.5, seed=7)
    right = build_partition(labels, num_clients=3, concentration=0.5, seed=8)

    assert left.client_indices != right.client_indices


def test_dirichlet_partition_avoids_empty_clients_when_samples_allow() -> None:
    from bbv.datasets.partitions import build_partition

    labels = [0, 0, 1, 1, 2, 2, 3, 3]

    result = build_partition(labels, num_clients=4, concentration=0.01, seed=1)

    assert all(count > 0 for count in result.client_sample_counts)


def test_split_metadata_round_trip(tmp_path: Path) -> None:
    metadata = SplitMetadata(
        dataset_name="cifar10",
        split_method="dirichlet_label_skew",
        seed=7,
        num_clients=3,
        concentration=0.5,
        total_samples=6,
        client_sample_counts=[2, 2, 2],
        client_indices=[[0, 3], [1, 4], [2, 5]],
        client_label_histograms=[
            {"0": 1, "1": 1},
            {"0": 1, "1": 1},
            {"0": 1, "1": 1},
        ],
        partition_params={"concentration": 0.5},
    )

    path = tmp_path / "split.json"
    save_split_metadata(path, metadata)

    loaded = load_split_metadata(path)

    assert loaded == metadata


def test_split_metadata_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"dataset_name": "cifar10"}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        load_split_metadata(path)


def test_saved_metadata_can_drive_subset_reconstruction(tmp_path: Path) -> None:
    from bbv.datasets.metadata import build_split_metadata
    from bbv.datasets.partitions import build_partition

    labels = [0, 1, 0, 1, 2, 2]
    partition = build_partition(labels, num_clients=3, concentration=0.5, seed=7)
    metadata = build_split_metadata("cifar10", 0.5, partition)
    path = tmp_path / "cifar10" / "split.json"
    save_split_metadata(path, metadata)

    loaded = load_split_metadata(path)
    reconstructed = [labels[index] for index in loaded.client_indices[0]]

    assert len(loaded.client_indices) == 3
    assert len(reconstructed) == loaded.client_sample_counts[0]


def test_load_dataset_returns_cifar10_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bbv.datasets.loaders import load_dataset

    captured: dict[str, object] = {}

    class FakeCIFAR10:
        def __init__(self, root: str, train: bool, download: bool) -> None:
            captured["root"] = root
            captured["train"] = train
            captured["download"] = download
            self.classes = ["airplane", "car", "bird"]

        def __len__(self) -> int:
            return 12

    monkeypatch.setattr(
        "bbv.datasets.loaders.datasets",
        SimpleNamespace(CIFAR10=FakeCIFAR10),
    )

    loaded = load_dataset(root=tmp_path / "raw", train=True, download=False)

    assert loaded.dataset_name == "cifar10"
    assert loaded.train is True
    assert loaded.num_classes == 3
    assert loaded.num_samples == 12
    assert captured == {
        "root": str(tmp_path / "raw"),
        "train": True,
        "download": False,
    }


def test_load_dataset_supports_cifar100(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bbv.datasets.loaders import load_dataset

    captured: dict[str, object] = {}

    class FakeCIFAR100:
        def __init__(self, root: str, train: bool, download: bool) -> None:
            captured["root"] = root
            captured["train"] = train
            captured["download"] = download
            self.classes = [str(index) for index in range(100)]

        def __len__(self) -> int:
            return 20

    monkeypatch.setattr(
        "bbv.datasets.loaders.datasets",
        SimpleNamespace(CIFAR100=FakeCIFAR100),
    )

    loaded = load_dataset(
        name="cifar100", root=tmp_path / "raw", train=False, download=False
    )

    assert loaded.dataset_name == "cifar100"
    assert loaded.split_name == "test"
    assert loaded.train is False
    assert loaded.num_classes == 100
    assert loaded.num_samples == 20
    assert captured == {
        "root": str(tmp_path / "raw"),
        "train": False,
        "download": False,
    }


def test_build_split_metadata_includes_partition_params() -> None:
    from bbv.datasets.metadata import build_split_metadata
    from bbv.datasets.partitions import build_partition

    partition = build_partition(
        [index % 4 for index in range(40)],
        num_clients=4,
        concentration=0.3,
        seed=5,
        partition_type="quantity_skew",
        quantity_sigma=1.0,
    )

    metadata = build_split_metadata("cifar10", 0.3, partition)

    assert metadata.partition_type == "quantity_skew"
    assert metadata.partition_params == {"concentration": 0.3, "quantity_sigma": 1.0}


def test_split_metadata_is_saved_under_expected_path(tmp_path: Path) -> None:
    from bbv.datasets.metadata import build_split_metadata
    from bbv.datasets.partitions import build_partition

    partition = build_partition([0, 1, 0, 1], num_clients=2, concentration=0.3, seed=7)
    metadata = build_split_metadata("cifar10", 0.3, partition)
    path = (
        tmp_path
        / "data"
        / "splits"
        / "cifar10"
        / "dirichlet_alpha-0.30_clients-2_seed-7.json"
    )

    save_split_metadata(path, metadata)

    assert path.exists()


def test_dirichlet_partition_rejects_invalid_client_count() -> None:
    from bbv.datasets.partitions import build_partition

    with pytest.raises(ValueError, match="num_clients"):
        build_partition([0, 1, 0], num_clients=0, concentration=0.5, seed=7)
