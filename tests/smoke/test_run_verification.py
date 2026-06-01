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


def test_run_verification_script_generates_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    repo_root = Path(__file__).resolve().parents[2]
    _prepare_femnist_fixture(repo_root)

    train_completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_watermark_baseline.py",
            f"output_root={output_root}",
            "dataset=femnist",
            "seed=0",
            "owner.id=owner0",
            "watermarking.code_length=8",
            "federated.rounds=1",
            "federated.num_clients=3",
            "federated.participation_rate=1.0",
            "federated.batch_size=2",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(repo_root),
    )
    assert train_completed.returncode == 0, train_completed.stderr

    verify_completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/run_verification.py",
            "dataset=femnist",
            "verification=margin",
            "owner.id=owner0",
            "seed=0",
            "verification.query_budget=4",
            "verification.batch_size=2",
            "verification.hard_label_only=false",
            f"output_root={output_root}",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(repo_root),
    )
    assert verify_completed.returncode == 0, verify_completed.stderr
    assert "Verification" in verify_completed.stderr

    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    latest = run_dirs[-1]
    summary = json.loads((latest / "verification_margin_summary.json").read_text())
    calibration = json.loads((latest / "calibration_artifacts.json").read_text())

    assert summary["owner_id"] == "owner0"
    assert "owner_score" in summary
    assert summary["query_budget"] == 4
    assert summary["hard_label_only"] is False
    assert "threshold" in calibration
    assert latest == max(run_dirs, key=lambda path: (path / "checkpoint.pt").stat().st_mtime if (path / "checkpoint.pt").exists() else path.stat().st_mtime)


def test_run_verification_script_rejects_owner_id_mismatch(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    repo_root = Path(__file__).resolve().parents[2]
    _prepare_femnist_fixture(repo_root)

    train_completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_watermark_baseline.py",
            f"output_root={output_root}",
            "dataset=femnist",
            "seed=0",
            "owner.id=owner0",
            "watermarking.code_length=8",
            "federated.rounds=1",
            "federated.num_clients=3",
            "federated.participation_rate=1.0",
            "federated.batch_size=2",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(repo_root),
    )
    assert train_completed.returncode == 0, train_completed.stderr

    verify_completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/run_verification.py",
            "dataset=femnist",
            "verification=margin",
            "owner.id=owner1",
            "seed=0",
            f"output_root={output_root}",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(repo_root),
    )

    assert verify_completed.returncode != 0


def test_run_verification_script_selects_latest_matching_owner_run(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    repo_root = Path(__file__).resolve().parents[2]
    _prepare_femnist_fixture(repo_root)

    for owner_id in ["owner0", "owner1"]:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/train/run_watermark_baseline.py",
                f"output_root={output_root}",
                "dataset=femnist",
                "seed=0",
                f"owner.id={owner_id}",
                "watermarking.code_length=8",
                "federated.rounds=1",
                "federated.num_clients=3",
                "federated.participation_rate=1.0",
                "federated.batch_size=2",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env=_script_env(repo_root),
        )
        assert completed.returncode == 0, completed.stderr

    verify_completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/run_verification.py",
            "dataset=femnist",
            "verification=margin",
            "owner.id=owner0",
            "seed=0",
            f"output_root={output_root}",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(repo_root),
    )
    assert verify_completed.returncode == 0, verify_completed.stderr

    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    matching = [
        path
        for path in run_dirs
        if json.loads((path / "owner_artifacts.json").read_text())["owner_id"] == "owner0"
    ]
    latest_matching = matching[-1]
    summary = json.loads((latest_matching / "verification_margin_summary.json").read_text())

    assert summary["owner_id"] == "owner0"


def test_run_verification_script_uses_latest_training_checkpoint_not_dir_mtime(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    repo_root = Path(__file__).resolve().parents[2]
    _prepare_femnist_fixture(repo_root)

    for _ in range(2):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/train/run_watermark_baseline.py",
                f"output_root={output_root}",
                "dataset=femnist",
                "seed=0",
                "owner.id=owner0",
                "watermarking.code_length=8",
                "federated.rounds=1",
                "federated.num_clients=3",
                "federated.participation_rate=1.0",
                "federated.batch_size=2",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env=_script_env(repo_root),
        )
        assert completed.returncode == 0, completed.stderr

    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    older_run, latest_run = run_dirs[0], run_dirs[-1]
    os.utime(older_run, None)

    verify_completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/run_verification.py",
            "dataset=femnist",
            "verification=margin",
            "owner.id=owner0",
            "seed=0",
            f"output_root={output_root}",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(repo_root),
    )
    assert verify_completed.returncode == 0, verify_completed.stderr

    assert not (older_run / "verification_margin_summary.json").exists()
    assert (latest_run / "verification_margin_summary.json").exists()
