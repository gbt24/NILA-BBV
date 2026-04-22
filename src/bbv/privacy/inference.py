from __future__ import annotations


def _compute_auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [score for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.5

    wins = 0.0
    total = len(positives) * len(negatives)
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total


def infer_label_skew_from_client_stats(
    *, client_stats: list[dict[str, float | int]], skew_targets: list[int]
) -> dict[str, float | list[float]]:
    if len(client_stats) != len(skew_targets):
        raise ValueError("client_stats and skew_targets must have the same length")
    if not client_stats:
        return {"auc": 0.5, "baseline_auc": 0.5, "scores": []}

    scores = [
        float(item.get("skew_ratio", 0.0))
        - 0.05 * float(item.get("class_coverage", 0.0))
        + 0.1 * float(item.get("main_wm_alignment", 0.0))
        - 0.1 * float(item.get("privacy_penalty", 0.0))
        for item in client_stats
    ]
    labels = [int(target) for target in skew_targets]
    auc = _compute_auc(scores, labels)
    return {
        "auc": float(auc),
        "baseline_auc": 0.5,
        "scores": [float(score) for score in scores],
    }
