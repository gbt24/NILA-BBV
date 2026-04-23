from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


LEAF_REPO_URL = "https://github.com/TalwalkarLab/leaf.git"


def clone_leaf_repo(cache_root: Path, *, force: bool = False) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    leaf_root = cache_root / "leaf"
    if force and leaf_root.exists():
        shutil.rmtree(leaf_root)
    if leaf_root.exists():
        return leaf_root
    subprocess.run(["git", "clone", LEAF_REPO_URL, str(leaf_root)], check=True)
    return leaf_root


def run_leaf_preprocess(leaf_root: Path, dataset_name: str) -> None:
    dataset_dir = leaf_root / "data" / dataset_name
    subprocess.run(["bash", "preprocess.sh", "-s", "niid", "-t", "sample"], cwd=dataset_dir, check=True)


def clean_leaf_dataset_cache(leaf_root: Path, dataset_name: str) -> None:
    dataset_data_dir = leaf_root / "data" / dataset_name / "data"
    for child_name in ["train", "test", "sampled_data", "rem_user_data", "all_data", "intermediate", "raw_data"]:
        child = dataset_data_dir / child_name
        if child.exists():
            shutil.rmtree(child)


def copy_prepared_leaf_data(*, leaf_root: Path, dataset_name: str, output_root: Path, force: bool = False) -> Path:
    source_dir = leaf_root / "data" / dataset_name / "data"
    if not (source_dir / "train").exists() or not (source_dir / "test").exists():
        raise FileNotFoundError(f"expected prepared LEAF data under {source_dir}")

    target_dir = output_root / dataset_name
    if force and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "test"]:
        split_target = target_dir / split_name
        if split_target.exists():
            shutil.rmtree(split_target)
        shutil.copytree(source_dir / split_name, split_target)
    return target_dir


def build_parser(dataset_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Prepare {dataset_name} in LEAF format")
    parser.add_argument("--output-root", default="data/raw", type=Path)
    parser.add_argument("--cache-root", default="data/cache", type=Path)
    parser.add_argument("--leaf-root", default=None, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main_prepare(dataset_name: str, preprocess_callback: object) -> int:
    parser = build_parser(dataset_name)
    args = parser.parse_args()
    leaf_root = args.leaf_root if args.leaf_root is not None else clone_leaf_repo(args.cache_root, force=False)
    if args.force:
        clean_leaf_dataset_cache(Path(leaf_root), dataset_name)
    prepared_source = Path(leaf_root) / "data" / dataset_name / "data"
    if not ((prepared_source / "train").exists() and (prepared_source / "test").exists()):
        preprocess_callback(Path(leaf_root), dataset_name)
    target_dir = copy_prepared_leaf_data(
        leaf_root=Path(leaf_root),
        dataset_name=dataset_name,
        output_root=args.output_root,
        force=bool(args.force),
    )
    print(f"Prepared {dataset_name} at {target_dir}")
    return 0


def entrypoint(dataset_name: str, preprocess_callback: object) -> None:
    raise SystemExit(main_prepare(dataset_name, preprocess_callback))
