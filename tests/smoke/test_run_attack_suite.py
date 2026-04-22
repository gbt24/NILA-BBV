from pathlib import Path
import subprocess
import sys


def test_attack_suite_script_generates_attacked_checkpoint(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
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

    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    checkpoint_path = run_dirs[-1] / "checkpoint.pt"

    attack_completed = subprocess.run(
        [
            sys.executable,
            "scripts/attacks/run_attack_suite.py",
            "attack=finetune",
            "dataset=cifar10",
            f"checkpoint={checkpoint_path}",
            "seed=0",
            f"output_root={tmp_path / 'attacks'}",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert attack_completed.returncode == 0, attack_completed.stderr

    attack_dirs = sorted(path for path in (tmp_path / "attacks").iterdir() if path.is_dir())
    latest = attack_dirs[-1]
    assert (latest / "attacked_checkpoint.pt").exists()
    assert (latest / "attack_log.json").exists()


def test_attack_suite_script_supports_extraction_attack(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
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

    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    checkpoint_path = run_dirs[-1] / "checkpoint.pt"

    attack_completed = subprocess.run(
        [
            sys.executable,
            "scripts/attacks/run_attack_suite.py",
            "attack=extraction",
            "dataset=cifar10",
            f"checkpoint={checkpoint_path}",
            "seed=0",
            f"output_root={tmp_path / 'attacks'}",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert attack_completed.returncode == 0, attack_completed.stderr
