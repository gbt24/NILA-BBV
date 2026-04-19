import json
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
    assert "threshold" in calibration
