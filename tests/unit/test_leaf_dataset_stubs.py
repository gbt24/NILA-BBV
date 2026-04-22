import pytest


def test_get_leaf_dataset_spec_returns_natural_split_stub() -> None:
    from bbv.datasets.leaf import get_leaf_dataset_spec

    spec = get_leaf_dataset_spec("femnist")

    assert spec.dataset_name == "femnist"
    assert spec.partition_type == "natural"
    assert spec.is_stub is True
    assert spec.num_classes == 62


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
        )

    monkeypatch.setattr("bbv.datasets.loaders.load_leaf_dataset", fake_load_leaf_dataset)

    loaded = load_dataset(name="femnist", root=tmp_path / "raw", train=True, download=False)

    assert loaded.dataset_name == "femnist"
    assert loaded.num_classes == 62
    assert loaded.num_samples == 1


def test_load_leaf_dataset_returns_stub_dataset_container(tmp_path) -> None:
    from bbv.datasets.leaf import load_leaf_dataset

    loaded = load_leaf_dataset(root=tmp_path / "raw", train=True, download=False, name="femnist")

    assert loaded.spec.dataset_name == "femnist"
    assert hasattr(loaded.dataset, "targets")
    sample, label = loaded.dataset[0]
    assert sample.shape == (3, 32, 32)
    assert isinstance(label, int)
