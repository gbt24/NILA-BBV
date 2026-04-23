from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_leaf_common import entrypoint, run_leaf_preprocess


def _prepare(leaf_root: Path, dataset_name: str) -> None:
    run_leaf_preprocess(leaf_root, dataset_name)


if __name__ == "__main__":
    entrypoint("sent140", _prepare)
