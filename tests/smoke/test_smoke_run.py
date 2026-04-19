import json
from pathlib import Path
import subprocess
import sys

import pytest

from bbv.federated.runner import run_smoke_experiment


def test_smoke_run_writes_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"

    result = run_smoke_experiment(output_root=output_dir)

    assert result.run_dir.exists()
    assert result.summary_path.exists()
    assert result.metrics["num_clients"] > 0
    assert result.metrics["final_loss"] >= 0.0

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["seed"] == 7
    assert summary["metrics"]["num_clients"] == 3
    assert summary["metrics"]["samples_per_client"] == 8


def test_smoke_run_rejects_invalid_client_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_clients"):
        run_smoke_experiment(output_root=tmp_path / "outputs", num_clients=0)


def test_smoke_script_runs_from_config(tmp_path: Path) -> None:
    output_root = tmp_path / "script-outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_smoke.py",
            f"output_root={output_root}",
            "seed=11",
            "federated.num_clients=2",
            "federated.samples_per_client=4",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summaries = sorted(output_root.glob("*/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["seed"] == 11
    assert summary["metrics"]["num_clients"] == 2
    assert summary["metrics"]["samples_per_client"] == 4
