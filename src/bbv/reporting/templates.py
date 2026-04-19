from __future__ import annotations


def build_tradeoff_svg(rows: list[dict[str, object]]) -> str:
    width = 640
    height = 320
    points: list[str] = []
    for index, row in enumerate(rows):
        x = 60 + index * 80
        owner_score = float(row.get("owner_score", 0.0))
        y = 260 - int(180 * max(0.0, min(1.0, owner_score)))
        points.append(f"{x},{y}")

    polyline = " ".join(points) if points else "60,260"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="#f7f8fb" />'
        '<text x="20" y="28" font-size="18" font-family="serif">Utility-Robustness Tradeoff</text>'
        '<line x1="60" y1="260" x2="600" y2="260" stroke="#333" stroke-width="1" />'
        '<line x1="60" y1="70" x2="60" y2="260" stroke="#333" stroke-width="1" />'
        f'<polyline points="{polyline}" fill="none" stroke="#1f6feb" stroke-width="3" />'
        '</svg>'
    )
