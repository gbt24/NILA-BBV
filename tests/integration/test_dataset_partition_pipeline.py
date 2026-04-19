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
