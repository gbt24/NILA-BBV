import json
import os
from pathlib import Path
import subprocess
import sys


def test_run_verification_script_generates_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    train_completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_watermark_baseline.py",
            f"output_root={output_root}",
            "seed=0",
            "owner.id=owner0",
            "watermarking.code_length=8",
            "federated.rounds=1",
            "federated.num_clients=3",
            "dataset.samples_per_client=12",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert train_completed.returncode == 0, train_completed.stderr

    verify_completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/run_verification.py",
            "dataset=cifar10",
            "verification=margin",
            "owner.id=owner0",
            "seed=0",
            "verification.query_budget=4",
            "verification.batch_size=2",
            "verification.hard_label_only=false",
            f"output_root={output_root}",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify_completed.returncode == 0, verify_completed.stderr

    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    latest = run_dirs[-1]
    summary = json.loads((latest / "verification_margin_summary.json").read_text())
    calibration = json.loads((latest / "calibration_artifacts.json").read_text())

    assert summary["owner_id"] == "owner0"
    assert "owner_score" in summary
    assert summary["query_budget"] == 4
    assert summary["hard_label_only"] is False
    assert "threshold" in calibration


def test_run_verification_script_rejects_owner_id_mismatch(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    train_completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_watermark_baseline.py",
            f"output_root={output_root}",
            "seed=0",
            "owner.id=owner0",
            "watermarking.code_length=8",
            "federated.rounds=1",
            "federated.num_clients=3",
            "dataset.samples_per_client=12",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert train_completed.returncode == 0, train_completed.stderr

    verify_completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/run_verification.py",
            "dataset=cifar10",
            "verification=margin",
            "owner.id=owner1",
            "seed=0",
            f"output_root={output_root}",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert verify_completed.returncode != 0


def test_run_verification_script_selects_latest_matching_owner_run(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    for owner_id in ["owner0", "owner1"]:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/train/run_watermark_baseline.py",
                f"output_root={output_root}",
                "seed=0",
                f"owner.id={owner_id}",
                "watermarking.code_length=8",
                "federated.rounds=1",
                "federated.num_clients=3",
                "dataset.samples_per_client=12",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    verify_completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/run_verification.py",
            "dataset=cifar10",
            "verification=margin",
            "owner.id=owner0",
            "seed=0",
            f"output_root={output_root}",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
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

    for _ in range(2):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/train/run_watermark_baseline.py",
                f"output_root={output_root}",
                "seed=0",
                "owner.id=owner0",
                "watermarking.code_length=8",
                "federated.rounds=1",
                "federated.num_clients=3",
                "dataset.samples_per_client=12",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    older_run, latest_run = run_dirs[0], run_dirs[-1]
    os.utime(older_run, None)

    verify_completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/run_verification.py",
            "dataset=cifar10",
            "verification=margin",
            "owner.id=owner0",
            "seed=0",
            f"output_root={output_root}",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify_completed.returncode == 0, verify_completed.stderr

    assert not (older_run / "verification_margin_summary.json").exists()
    assert (latest_run / "verification_margin_summary.json").exists()
