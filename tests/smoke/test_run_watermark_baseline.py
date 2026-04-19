import json
from pathlib import Path
import subprocess
import sys


def test_run_watermark_baseline_generates_artifacts_and_verification(tmp_path: Path) -> None:
    output_root = tmp_path / "watermark-outputs"

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
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    artifacts = json.loads((run_dir / "owner_artifacts.json").read_text())
    verification = json.loads((run_dir / "verification_summary.json").read_text())

    assert artifacts["owner_id"] == "owner0"
    assert len(artifacts["codebook"]) == 8
    assert verification["owner_id"] == "owner0"
