import json
from pathlib import Path

from bbv.federated import train_federated


def _write_sent140_split(root: Path, split_name: str, payload: dict[str, object]) -> None:
    split_dir = root / "sent140" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "all_data_0.json").write_text(json.dumps(payload), encoding="utf-8")


def test_train_federated_runs_on_sent140(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_sent140_split(
        root,
        "train",
        {
            "users": ["user0", "user1"],
            "num_samples": [2, 2],
            "user_data": {
                "user0": {
                    "x": [
                        ["0", "0", "0", "user0", "great movie"],
                        ["0", "0", "0", "user0", "bad acting"],
                    ],
                    "y": [4, 0],
                },
                "user1": {
                    "x": [
                        ["0", "0", "0", "user1", "nice plot"],
                        ["0", "0", "0", "user1", "terrible ending"],
                    ],
                    "y": [4, 0],
                },
            },
        },
    )
    _write_sent140_split(
        root,
        "test",
        {
            "users": ["user2"],
            "num_samples": [2],
            "user_data": {
                "user2": {
                    "x": [
                        ["0", "0", "0", "user2", "awesome finale"],
                        ["0", "0", "0", "user2", "awful sequel"],
                    ],
                    "y": [4, 0],
                },
            },
        },
    )

    from bbv.datasets.loaders import load_dataset

    def load_from_tmp(*, name: str, root: Path, train: bool, download: bool):
        del root, download
        return load_dataset(name=name, root=tmp_path / "raw", train=train, download=False)

    monkeypatch.setattr("bbv.federated.fedavg.load_dataset", load_from_tmp, raising=False)

    result = train_federated(
        output_root=tmp_path / "outputs",
        seed=0,
        dataset_name="sent140",
        model_name="text_cnn",
        num_classes=2,
        num_clients=2,
        rounds=1,
        participation_rate=1.0,
        local_epochs=1,
        batch_size=2,
        learning_rate=0.01,
        samples_per_client=2,
        partition_type="natural",
    )

    assert result.output_dir.exists()
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "sent140"
    assert payload["partition"]["partition_type"] == "natural"
