from bbv.verification import compute_owner_score


def test_compute_owner_score_matches_hamming_formula() -> None:
    score = compute_owner_score(
        expected_codebook=[0, 1, 0, 1],
        recovered_codebook=[0, 1, 1, 1],
        negative_asr=0.25,
        negative_weight=0.2,
    )
    assert score == 1.0 - (1 / 4) - 0.2 * 0.25
