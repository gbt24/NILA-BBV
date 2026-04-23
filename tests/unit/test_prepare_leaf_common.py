from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))

from prepare_leaf_common import clean_leaf_dataset_cache


def test_clean_leaf_dataset_cache_preserves_raw_data_archives(tmp_path: Path) -> None:
    leaf_root = tmp_path / "leaf"
    dataset_data_dir = leaf_root / "data" / "femnist" / "data"

    train_dir = dataset_data_dir / "train"
    test_dir = dataset_data_dir / "test"
    intermediate_dir = dataset_data_dir / "intermediate"
    raw_data_dir = dataset_data_dir / "raw_data"

    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    intermediate_dir.mkdir(parents=True)
    raw_data_dir.mkdir(parents=True)

    (raw_data_dir / "by_class.zip").write_bytes(b"zip-content")
    (raw_data_dir / "by_write.zip").write_bytes(b"zip-content")

    clean_leaf_dataset_cache(leaf_root, "femnist")

    assert not train_dir.exists()
    assert not test_dir.exists()
    assert not intermediate_dir.exists()
    assert (raw_data_dir / "by_class.zip").exists()
    assert (raw_data_dir / "by_write.zip").exists()
