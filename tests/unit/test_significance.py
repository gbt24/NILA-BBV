from bbv.evaluation.significance import exact_binomial_ci


def test_exact_binomial_ci_is_ordered() -> None:
    lower, upper = exact_binomial_ci(successes=9, total=10, alpha=0.05)
    assert 0.0 <= lower <= upper <= 1.0
