from pathlib import Path
import subprocess
import sys


def test_build_report_full_exports_tradeoff_figure(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "verification_margin_summary.json").write_text(
        """{
  \"owner_id\": \"owner0\",
  \"owner_score\": 0.71,
  \"decision\": true,
  \"threshold\": 0.5,
  \"margin_value\": 0.2,
  \"competitor_scores\": {\"owner1\": 0.2},
  \"ambiguity_flag\": false
}\n""",
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text('{"seed": 0}\n', encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report/build_report.py",
            "dataset=cifar10",
            "study=main",
            f"outputs_dir={tmp_path / 'runs'}",
            f"report_root={tmp_path / 'outputs'}",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "outputs" / "figures" / "cifar10-main-tradeoff-figure.svg").exists()
