from pathlib import Path


def test_partition_pipeline_supports_shard_and_metadata(tmp_path: Path) -> None:
    from bbv.datasets.metadata import build_split_metadata, load_split_metadata, save_split_metadata
    from bbv.datasets.partitions import build_partition

    labels = [index % 4 for index in range(40)]
    partition = build_partition(
        labels,
        num_clients=5,
        concentration=0.5,
        seed=13,
        partition_type="shard",
        shards_per_client=2,
    )
    metadata = build_split_metadata("cifar10", 0.5, partition)
    path = tmp_path / "data" / "splits" / "cifar10" / "shard-split-seed13.json"
    save_split_metadata(path, metadata)
    loaded = load_split_metadata(path)

    assert partition.partition_type == "shard"
    assert sum(partition.client_sample_counts) == len(labels)
    assert loaded.split_method == "shard"
    assert loaded.total_samples == len(labels)


def test_build_client_datasets_returns_subset_and_label_histogram() -> None:
    import torch

    from bbv.datasets.loaders import LoadedDataset
    from bbv.datasets.partitions import build_partition
    from bbv.federated.client_data import build_client_datasets

    class FakeDataset:
        def __init__(self) -> None:
            self.classes = ["0", "1"]
            self.targets = [0, 1, 0, 1, 0, 1]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), int(self.targets[index])

    loaded_dataset = LoadedDataset(
        dataset_name="cifar10",
        split_name="train",
        train=True,
        num_classes=2,
        num_samples=6,
        dataset=FakeDataset(),
    )
    partition = build_partition(
        [0, 1, 0, 1, 0, 1],
        num_clients=2,
        concentration=0.5,
        seed=7,
    )

    client_datasets = build_client_datasets(loaded_dataset, partition)

    assert len(client_datasets) == 2
    assert hasattr(client_datasets[0].dataset, "__getitem__")
    assert client_datasets[0].label_histogram == partition.client_label_histograms[0]
