from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bbv.federated.progress import progress_iterable


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LEAF datasets for bbv")
    parser.add_argument("--dataset", choices=["femnist", "sent140", "all"], default="all")
    parser.add_argument("--output-root", default="data/raw", type=Path)
    parser.add_argument("--cache-root", default="data/cache", type=Path)
    parser.add_argument("--leaf-root", default=None, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    scripts = []
    if args.dataset in {"femnist", "all"}:
        scripts.append(repo_root / "scripts" / "data" / "prepare_femnist.py")
    if args.dataset in {"sent140", "all"}:
        scripts.append(repo_root / "scripts" / "data" / "prepare_sent140.py")

    base_args = [f"--output-root={args.output_root}", f"--cache-root={args.cache_root}"]
    if args.leaf_root is not None:
        base_args.append(f"--leaf-root={args.leaf_root}")
    if args.force:
        base_args.append("--force")

    for script in progress_iterable(
        scripts,
        description="Preparing datasets",
        enabled=not args.no_progress,
        leave=True,
    ):
        subprocess.run([sys.executable, str(script), *base_args], cwd=repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
