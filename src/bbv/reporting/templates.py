from __future__ import annotations


def build_tradeoff_svg(rows: list[dict[str, object]]) -> str:
    width = 640
    height = 320
    points: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        margin = max(0.0, min(1.0, float(row.get("margin_value", 0.0))))
        owner_score = float(row.get("owner_score", 0.0))
        x = 60 + int(520 * margin)
        y = 260 - int(180 * max(0.0, min(1.0, owner_score)))
        points.append((x, y))

    polyline = " ".join(f"{x},{y}" for x, y in points) if points else "60,260"
    markers = "".join(
        f'<circle cx="{x}" cy="{y}" r="4" fill="#1f6feb" />' for x, y in points
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="#f7f8fb" />'
        '<text x="20" y="28" font-size="18" font-family="serif">Utility-Robustness Tradeoff</text>'
        '<line x1="60" y1="260" x2="600" y2="260" stroke="#333" stroke-width="1" />'
        '<line x1="60" y1="70" x2="60" y2="260" stroke="#333" stroke-width="1" />'
        '<text x="330" y="295" text-anchor="middle" font-size="12">margin</text>'
        '<text x="18" y="165" text-anchor="middle" font-size="12" transform="rotate(-90 18 165)">owner score</text>'
        f'<polyline points="{polyline}" fill="none" stroke="#1f6feb" stroke-width="3" />'
        f"{markers}"
        '</svg>'
    )


def build_score_distribution_svg(rows: list[dict[str, object]]) -> str:
    width = 640
    height = 320
    cohorts = {
        "owner": [],
        "false_claim": [],
        "non_owner": [],
    }
    for row in rows:
        claim_type = str(row.get("claim_type", "owner")).lower()
        score = max(0.0, min(1.0, float(row.get("owner_score", 0.0))))
        if claim_type not in cohorts:
            claim_type = "non_owner"
        cohorts[claim_type].append(score)

    labels = ["owner", "false_claim", "non_owner"]
    fills = ["#2a7f62", "#d97706", "#1f6feb"]
    bars: list[str] = []
    for index, label in enumerate(labels):
        values = cohorts[label]
        mean_score = sum(values) / len(values) if values else 0.0
        x = 90 + index * 160
        bar_height = int(180 * mean_score)
        y = 250 - bar_height
        bars.append(f'<rect x="{x}" y="{y}" width="90" height="{bar_height}" fill="{fills[index]}" />')
        bars.append(f'<text x="{x + 45}" y="274" text-anchor="middle" font-size="12">{label}</text>')
        bars.append(f'<text x="{x + 45}" y="{y - 8}" text-anchor="middle" font-size="11">{mean_score:.2f}</text>')
        bars.append(f'<text x="{x + 45}" y="294" text-anchor="middle" font-size="10">n={len(values)}</text>')

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="#f8faf8" />'
        '<text x="20" y="28" font-size="18" font-family="serif">Owner / Non-owner Score Distribution</text>'
        '<line x1="60" y1="250" x2="600" y2="250" stroke="#333" stroke-width="1" />'
        '<line x1="60" y1="70" x2="60" y2="250" stroke="#333" stroke-width="1" />'
        '<text x="18" y="165" text-anchor="middle" font-size="12" transform="rotate(-90 18 165)">mean owner score</text>'
        + "".join(bars)
        + '</svg>'
    )
