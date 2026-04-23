from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_leaf_common import entrypoint, run_leaf_preprocess


def _patch_leaf_femnist_pillow_compat(leaf_root: Path) -> None:
    target = leaf_root / "data" / "femnist" / "preprocess" / "data_to_json.py"
    if not target.exists():
        return
    text = target.read_text(encoding="utf-8")
    updated = text.replace("Image.ANTIALIAS", "Image.Resampling.LANCZOS")
    if updated != text:
        target.write_text(updated, encoding="utf-8")


def _prepare(leaf_root: Path, dataset_name: str) -> None:
    _patch_leaf_femnist_pillow_compat(leaf_root)
    run_leaf_preprocess(leaf_root, dataset_name)


if __name__ == "__main__":
    entrypoint("femnist", _prepare)
