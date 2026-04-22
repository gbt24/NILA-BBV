import json
import subprocess
import sys
from pathlib import Path


def test_run_watermark_baseline_writes_owner_commitment(tmp_path: Path) -> None:
    output_root = tmp_path / "watermark-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_watermark_baseline.py",
            f"output_root={output_root}",
            "seed=7",
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
    commitment_path = run_dir / "owner_commitment.json"
    assert commitment_path.exists()

    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    assert commitment["owner_id"] == "owner0"
    assert commitment["seed"] == 7
    assert "timestamp" in commitment
    assert "codebook_hash" in commitment
