import json
from pathlib import Path
import subprocess
import sys


def _write_leaf_shard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "users": ["user0"],
                "num_samples": [1],
                "user_data": {"user0": {"x": [[0.0] * 784], "y": [0]}},
            }
        ),
        encoding="utf-8",
    )


def test_prepare_leaf_datasets_script_copies_from_existing_leaf_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    leaf_root = tmp_path / "leaf"
    _write_leaf_shard(leaf_root / "data" / "femnist" / "data" / "train" / "all_data_0.json")
    _write_leaf_shard(leaf_root / "data" / "femnist" / "data" / "test" / "all_data_0.json")
    _write_leaf_shard(leaf_root / "data" / "sent140" / "data" / "train" / "all_data_0.json")
    _write_leaf_shard(leaf_root / "data" / "sent140" / "data" / "test" / "all_data_0.json")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/data/prepare_leaf_datasets.py",
            "--dataset=all",
            f"--leaf-root={leaf_root}",
            f"--output-root={tmp_path / 'raw'}",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "raw" / "femnist" / "train" / "all_data_0.json").exists()
    assert (tmp_path / "raw" / "sent140" / "train" / "all_data_0.json").exists()
