from pathlib import Path

from bbv.watermarking.baseline import (
    build_positive_queries,
    generate_codebook,
    load_owner_artifacts,
    save_owner_artifacts,
)


def test_generate_codebook_is_reproducible_and_unique() -> None:
    left = generate_codebook(owner_id="owner0", code_length=16, seed=7)
    right = generate_codebook(owner_id="owner0", code_length=16, seed=7)
    other = generate_codebook(owner_id="owner1", code_length=16, seed=7)

    assert left == right
    assert left != other
    assert len(left) == 16
    assert set(left).issubset({0, 1})


def test_positive_queries_follow_codebook_length() -> None:
    codebook = generate_codebook(owner_id="owner0", code_length=12, seed=3)
    queries = build_positive_queries(codebook=codebook, seed=3)

    assert len(queries) == 12
    assert all(sample.shape == (3, 32, 32) for sample in queries)


def test_owner_artifacts_round_trip(tmp_path: Path) -> None:
    codebook = generate_codebook(owner_id="owner0", code_length=10, seed=11)
    queries = build_positive_queries(codebook=codebook, seed=11)
    path = tmp_path / "owner0" / "artifacts.json"

    save_owner_artifacts(path=path, owner_id="owner0", codebook=codebook, queries=queries)
    loaded = load_owner_artifacts(path)

    assert loaded["owner_id"] == "owner0"
    assert loaded["codebook"] == codebook
    assert len(loaded["positive_queries"]) == len(codebook)
