import json
from pathlib import Path
import subprocess
import sys


def test_run_fedavg_script_generates_run_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "fedavg-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_fedavg.py",
            f"output_root={output_root}",
            "seed=0",
            "federated.rounds=1",
            "federated.num_clients=3",
            "federated.participation_rate=0.67",
            "federated.local_epochs=1",
            "federated.batch_size=8",
            "federated.learning_rate=0.05",
            "dataset.samples_per_client=12",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"] == "cifar10"
    assert metrics["model"] == "resnet18"
    assert len(metrics["rounds"]) == 1


def test_run_fedavg_script_emits_progress_output_when_enabled(tmp_path: Path) -> None:
    output_root = tmp_path / "fedavg-progress-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_fedavg.py",
            f"output_root={output_root}",
            "seed=0",
            "federated.rounds=1",
            "federated.num_clients=3",
            "federated.participation_rate=0.67",
            "federated.local_epochs=1",
            "federated.batch_size=8",
            "federated.learning_rate=0.05",
            "dataset.samples_per_client=12",
            "progress.enabled=true",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Rounds" in completed.stderr
    assert "Clients" in completed.stderr
