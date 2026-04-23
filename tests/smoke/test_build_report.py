from pathlib import Path
import subprocess
import sys


def test_build_report_script_exports_main_outputs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs" / "demo-run"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "verification_margin_summary.json").write_text(
        """{
  \"owner_id\": \"owner0\",
  \"owner_score\": 0.68,
  \"decision\": true,
  \"threshold\": 0.5,
  \"margin_value\": 0.15,
  \"competitor_scores\": {\"owner1\": 0.25}
}\n""",
        encoding="utf-8",
    )
    (runs_dir / "run_metadata.json").write_text(
        '{"allocation": {"enabled": false}, "seed": 0}\n', encoding="utf-8"
    )

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
    assert "Report build" in completed.stderr

    assert (tmp_path / "outputs" / "tables" / "cifar10-main-main-results.csv").exists()
    assert (tmp_path / "outputs" / "figures" / "cifar10-main-main-figure.svg").exists()
    assert (tmp_path / "outputs" / "summaries" / "cifar10-main-summary.md").exists()
