import json
import os
from pathlib import Path
import subprocess
import sys


def _write_femnist_split(root: Path, split_name: str, payload: dict[str, object]) -> None:
    split_dir = root / "femnist" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "all_data_0.json").write_text(json.dumps(payload), encoding="utf-8")


def _prepare_femnist_fixture(repo_root: Path) -> None:
    raw_root = repo_root / "data" / "raw"
    _write_femnist_split(
        raw_root,
        "train",
        {
            "users": ["writer0", "writer1", "writer2"],
            "num_samples": [2, 2, 2],
            "user_data": {
                "writer0": {"x": [[0.0] * 784, [1.0] * 784], "y": [0, 1]},
                "writer1": {"x": [[0.25] * 784, [0.5] * 784], "y": [1, 2]},
                "writer2": {"x": [[0.75] * 784, [0.9] * 784], "y": [2, 3]},
            },
        },
    )
    _write_femnist_split(
        raw_root,
        "test",
        {
            "users": ["writer3"],
            "num_samples": [2],
            "user_data": {
                "writer3": {"x": [[0.1] * 784, [0.6] * 784], "y": [4, 5]},
            },
        },
    )


def _script_env(repo_root: Path) -> dict[str, str]:
    pythonpath = str(repo_root / "src")
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath = f"{pythonpath}{os.pathsep}{existing}"
    return {**os.environ, "PYTHONPATH": pythonpath}


def test_run_fedavg_script_generates_run_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "fedavg-outputs"
    repo_root = Path(__file__).resolve().parents[2]
    _prepare_femnist_fixture(repo_root)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_fedavg.py",
            f"output_root={output_root}",
            "dataset=femnist",
            "seed=0",
            "federated.rounds=1",
            "federated.num_clients=3",
            "federated.participation_rate=1.0",
            "federated.local_epochs=1",
            "federated.batch_size=2",
            "federated.learning_rate=0.05",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(repo_root),
    )

    assert completed.returncode == 0, completed.stderr
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"] == "femnist"
    assert metrics["model"] == "resnet18"
    assert len(metrics["rounds"]) == 1


def test_run_fedavg_script_emits_progress_output_when_enabled(tmp_path: Path) -> None:
    output_root = tmp_path / "fedavg-progress-outputs"
    repo_root = Path(__file__).resolve().parents[2]
    _prepare_femnist_fixture(repo_root)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_fedavg.py",
            f"output_root={output_root}",
            "dataset=femnist",
            "seed=0",
            "federated.rounds=1",
            "federated.num_clients=3",
            "federated.participation_rate=1.0",
            "federated.local_epochs=1",
            "federated.batch_size=2",
            "federated.learning_rate=0.05",
            "progress.enabled=true",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(repo_root),
    )

    assert completed.returncode == 0, completed.stderr
    assert "Rounds" in completed.stderr
    assert "Clients" in completed.stderr
