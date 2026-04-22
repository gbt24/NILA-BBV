import pytest


def test_get_leaf_dataset_spec_marks_femnist_as_real_natural_dataset() -> None:
    from bbv.datasets.leaf import get_leaf_dataset_spec

    spec = get_leaf_dataset_spec("femnist")

    assert spec.dataset_name == "femnist"
    assert spec.partition_type == "natural"
    assert spec.is_stub is False
    assert spec.num_classes == 62


def test_get_leaf_dataset_spec_keeps_shakespeare_stub() -> None:
    from bbv.datasets.leaf import get_leaf_dataset_spec

    spec = get_leaf_dataset_spec("shakespeare")

    assert spec.dataset_name == "shakespeare"
    assert spec.partition_type == "natural"
    assert spec.is_stub is True


def test_get_leaf_dataset_spec_rejects_unsupported_dataset() -> None:
    from bbv.datasets.leaf import get_leaf_dataset_spec

    with pytest.raises(ValueError, match="unsupported LEAF dataset"):
        get_leaf_dataset_spec("cifar10")


def test_load_dataset_dispatches_leaf_stub(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from bbv.datasets.loaders import load_dataset

    def fake_load_leaf_dataset(*, root, train, download, name):
        return SimpleNamespace(
            spec=SimpleNamespace(num_classes=62),
            split_name="train" if train else "test",
            train=train,
            dataset=[("stub", 0)],
            client_indices=[[0]],
            user_ids=["writer0"],
        )

    monkeypatch.setattr("bbv.datasets.loaders.load_leaf_dataset", fake_load_leaf_dataset)

    loaded = load_dataset(name="femnist", root=tmp_path / "raw", train=True, download=False)

    assert loaded.dataset_name == "femnist"
    assert loaded.num_classes == 62
    assert loaded.num_samples == 1


def test_load_leaf_dataset_requires_real_femnist_layout(tmp_path) -> None:
    from bbv.datasets.leaf import load_leaf_dataset

    with pytest.raises(FileNotFoundError, match="expected FEMNIST split directory"):
        load_leaf_dataset(root=tmp_path / "raw", train=True, download=False, name="femnist")


def test_load_leaf_dataset_reads_real_femnist_layout(tmp_path) -> None:
    import json

    from bbv.datasets.leaf import load_leaf_dataset

    root = tmp_path / "raw"
    train_dir = root / "femnist" / "train"
    test_dir = root / "femnist" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    train_payload = {
        "users": ["writer0", "writer1"],
        "num_samples": [2, 2],
        "user_data": {
            "writer0": {"x": [[0.0] * 784, [1.0] * 784], "y": [0, 1]},
            "writer1": {"x": [[0.5] * 784, [0.25] * 784], "y": [1, 2]},
        },
    }
    test_payload = {
        "users": ["writer2"],
        "num_samples": [1],
        "user_data": {
            "writer2": {"x": [[0.75] * 784], "y": [3]},
        },
    }

    (train_dir / "all_data_0.json").write_text(json.dumps(train_payload), encoding="utf-8")
    (test_dir / "all_data_0.json").write_text(json.dumps(test_payload), encoding="utf-8")

    loaded = load_leaf_dataset(root=root, train=True, download=False, name="femnist")

    assert loaded.spec.dataset_name == "femnist"
    assert loaded.spec.is_stub is False
    assert loaded.client_indices == [[0, 1], [2, 3]]
    sample, label = loaded.dataset[0]
    assert sample.shape == (3, 32, 32)
    assert label == 0
