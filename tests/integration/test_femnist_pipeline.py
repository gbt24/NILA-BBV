import json
from pathlib import Path

from bbv.datasets.loaders import load_dataset
from bbv.federated import train_federated


def _write_femnist_split(root: Path, split_name: str, payload: dict[str, object]) -> None:
    split_dir = root / "femnist" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "all_data_0.json").write_text(json.dumps(payload), encoding="utf-8")


def test_train_federated_runs_on_femnist_natural_split(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_femnist_split(
        root,
        "train",
        {
            "users": ["writer0", "writer1"],
            "num_samples": [2, 2],
            "user_data": {
                "writer0": {"x": [[0.0] * 784, [1.0] * 784], "y": [0, 1]},
                "writer1": {"x": [[0.5] * 784, [0.25] * 784], "y": [1, 2]},
            },
        },
    )
    _write_femnist_split(
        root,
        "test",
        {
            "users": ["writer2"],
            "num_samples": [2],
            "user_data": {
                "writer2": {"x": [[0.75] * 784, [0.9] * 784], "y": [3, 4]},
            },
        },
    )

    def load_from_tmp(*, name: str, root: Path, train: bool, download: bool):
        del root, download
        return load_dataset(name=name, root=tmp_path / "raw", train=train, download=False)

    monkeypatch.setattr("bbv.federated.fedavg.load_dataset", load_from_tmp, raising=False)

    result = train_federated(
        output_root=tmp_path / "outputs",
        seed=0,
        dataset_name="femnist",
        model_name="resnet18",
        num_classes=62,
        num_clients=2,
        rounds=1,
        participation_rate=1.0,
        local_epochs=1,
        batch_size=2,
        learning_rate=0.01,
        samples_per_client=2,
        partition_type="natural",
    )

    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "femnist"
    assert payload["partition"]["partition_type"] == "natural"
    assert payload["partition"]["client_sample_counts"] == [2, 2]
