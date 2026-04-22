def test_build_partition_supports_quantity_skew() -> None:
    from bbv.datasets.partitions import build_partition

    labels = [i % 10 for i in range(100)]
    result = build_partition(
        labels,
        num_clients=10,
        concentration=0.3,
        seed=0,
        partition_type="quantity_skew",
        quantity_sigma=1.0,
    )

    assert result.partition_type == "quantity_skew"
    assert max(result.client_sample_counts) != min(result.client_sample_counts)


def test_build_partition_supports_combined_label_quantity() -> None:
    from bbv.datasets.partitions import build_partition

    labels = [i % 5 for i in range(100)]
    result = build_partition(
        labels,
        num_clients=10,
        concentration=0.2,
        seed=3,
        partition_type="combined_label_quantity",
        quantity_sigma=0.8,
    )

    assert result.partition_type == "combined_label_quantity"
    assert sum(result.client_sample_counts) == len(labels)
    assert max(result.client_sample_counts) != min(result.client_sample_counts)
    assert any(len(histogram) < 5 for histogram in result.client_label_histograms)
