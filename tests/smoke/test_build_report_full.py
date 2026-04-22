from pathlib import Path
import subprocess
import sys


def test_task10_config_surface_and_report_bundle_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    required_configs = [
        repo_root / "configs" / "eval" / "dataset" / "cifar100.yaml",
        repo_root / "configs" / "attacks" / "attack" / "finetune.yaml",
        repo_root / "configs" / "attacks" / "attack" / "pruning.yaml",
        repo_root / "configs" / "attacks" / "attack" / "quantization.yaml",
        repo_root / "configs" / "attacks" / "attack" / "distillation.yaml",
        repo_root / "configs" / "attacks" / "attack" / "extraction.yaml",
        repo_root / "configs" / "report" / "dataset" / "cifar10.yaml",
        repo_root / "configs" / "report" / "dataset" / "cifar100.yaml",
        repo_root / "configs" / "report" / "study" / "main.yaml",
    ]

    for config_path in required_configs:
        assert config_path.exists(), f"missing Task 10 config: {config_path.relative_to(repo_root)}"

    config_checks = [
        [
            sys.executable,
            "scripts/train/run_watermark_baseline.py",
            "--cfg",
            "job",
            "dataset=cifar100",
            "allocation=adaptive",
        ],
        [
            sys.executable,
            "scripts/eval/run_verification.py",
            "--cfg",
            "job",
            "dataset=cifar100",
            "verification=margin",
        ],
    ]
    for command in config_checks:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    for attack_name in ["finetune", "pruning", "quantization", "distillation", "extraction"]:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/attacks/run_attack_suite.py",
                "--cfg",
                "job",
                f"attack={attack_name}",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_task10_attack_default_checkpoint_resolves_latest_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    attacks_root = tmp_path / "attacks"

    train_completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/run_watermark_baseline.py",
            "seed=0",
            "owner.id=owner0",
            "watermarking.code_length=8",
            "federated.rounds=1",
            "federated.num_clients=3",
            "dataset.samples_per_client=12",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert train_completed.returncode == 0, train_completed.stderr

    attack_completed = subprocess.run(
        [
            sys.executable,
            "scripts/attacks/run_attack_suite.py",
            "attack=finetune",
            "dataset=cifar10",
            f"output_root={attacks_root}",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert attack_completed.returncode == 0, attack_completed.stderr
    assert any(path.is_dir() for path in attacks_root.iterdir())

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

    bundle_dir = tmp_path / "bundles" / "task10" / "cifar100-main"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report/build_report.py",
            "dataset=cifar100",
            "study=main",
            f"outputs_dir={tmp_path / 'runs'}",
            f"bundle_dir={bundle_dir}",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (bundle_dir / "figures" / "cifar100-main-tradeoff-figure.svg").exists()
    assert (bundle_dir / "tables" / "cifar100-main-main-results.csv").exists()
    assert (bundle_dir / "summaries" / "cifar100-main-summary.md").exists()


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
    assert (tmp_path / "outputs" / "figures" / "cifar10-main-main-figure.svg").exists()
    assert (tmp_path / "outputs" / "tables" / "cifar10-main-main-results.csv").exists()
    assert (tmp_path / "outputs" / "tables" / "cifar10-main-ablation-results.csv").exists()
    assert (tmp_path / "outputs" / "tables" / "cifar10-main-robustness-results.csv").exists()
    summary_text = (tmp_path / "outputs" / "summaries" / "cifar10-main-summary.md").read_text(
        encoding="utf-8"
    )
    assert "acceptance_rate" in summary_text
    assert "ambiguity_rate" in summary_text
    assert "fpr" in summary_text
    assert "fnr" in summary_text
    assert "false_claim_acceptance_rate" in summary_text
    assert "robustness_acceptance_rate" in summary_text
