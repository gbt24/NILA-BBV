import json
import os
from pathlib import Path
import subprocess
import sys


def _script_env(repo_root: Path) -> dict[str, str]:
    pythonpath = str(repo_root / "src")
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath = f"{pythonpath}{os.pathsep}{existing}"
    return {**os.environ, "PYTHONPATH": pythonpath}


def _write_summary(path: Path, owner_score: float, *, owner_id: str = "owner0") -> None:
    path.write_text(
        json.dumps(
            {
                "owner_id": owner_id,
                "owner_score": owner_score,
                "competitor_scores": {"owner1": 0.52, "owner2": 0.61},
                "query_budget": 64,
                "hard_label_only": True,
            }
        ),
        encoding="utf-8",
    )


def test_run_false_claim_protocol_writes_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    owner_summary = tmp_path / "owner.json"
    calibration_owner = tmp_path / "cal_owner.json"
    calibration_non_owner_a = tmp_path / "cal_non_owner_a.json"
    calibration_non_owner_b = tmp_path / "cal_non_owner_b.json"
    false_claim = tmp_path / "false_claim.json"
    output_path = tmp_path / "protocol.json"

    _write_summary(owner_summary, 0.86)
    _write_summary(calibration_owner, 0.84)
    _write_summary(calibration_non_owner_a, 0.3)
    _write_summary(calibration_non_owner_b, 0.55)
    _write_summary(false_claim, 0.72, owner_id="owner9")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/run_false_claim_protocol.py",
            f"owner_summary_path={owner_summary}",
            f"calibration_owner_summary_paths=[{calibration_owner}]",
            f"calibration_non_owner_summary_paths=[{calibration_non_owner_a},{calibration_non_owner_b}]",
            f"false_claim_summary_paths=[{false_claim}]",
            "target_fpr=0.5",
            "margin=0.1",
            f"output_path={output_path}",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(repo_root),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["owner_id"] == "owner0"
    assert payload["false_claim_trial_count"] == 1
    assert payload["non_owner_calibration_sample_count"] == 2
    assert "calibrated_threshold" in payload
