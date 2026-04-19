from pathlib import Path

import torch

from bbv.federated.fedavg import train_federated


def test_train_federated_writes_metrics_checkpoint_and_metadata(tmp_path: Path) -> None:
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

    assert result.run_dir.exists()
    assert result.metrics_path.exists()
    assert result.metadata_path.exists()
    assert result.checkpoint_path.exists()

    payload = result.metrics_path.read_text(encoding="utf-8")
    assert '"round": 1' in payload
    assert '"round": 2' in payload

    state = torch.load(result.checkpoint_path, map_location="cpu")
    assert "model_state" in state
