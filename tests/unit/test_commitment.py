from bbv.watermarking.commitment import build_commitment_record


def test_build_commitment_record_contains_seed_hash_timestamp() -> None:
    record = build_commitment_record(
        owner_id="owner0",
        seed=7,
        codebook=[0, 1, 1, 0],
        config={"wm_weight": 0.2},
    )

    assert "timestamp" in record
    assert "codebook_hash" in record
    assert "config_hash" in record
    assert record["seed"] == 7
