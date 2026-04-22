"""Lightweight owner commitment helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def build_codebook_hash(codebook: list[int]) -> str:
    return hashlib.sha256("".join(str(bit) for bit in codebook).encode("utf-8")).hexdigest()


def build_config_hash(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_commitment_record(
    *,
    owner_id: str,
    seed: int,
    codebook: list[int],
    config: dict[str, object],
) -> dict[str, object]:
    return {
        "owner_id": owner_id,
        "seed": seed,
        "codebook_hash": build_codebook_hash(codebook),
        "timestamp": datetime.now(UTC).isoformat(),
        "config_hash": build_config_hash(config),
    }


def save_commitment_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
