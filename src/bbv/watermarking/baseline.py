"""Minimal watermark baseline artifacts for Phase 3."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch


def generate_codebook(owner_id: str, code_length: int, seed: int) -> list[int]:
    if code_length <= 0:
        raise ValueError("code_length must be greater than 0")
    owner_offset = sum(ord(ch) for ch in owner_id)
    rng = np.random.default_rng(seed + owner_offset)
    return rng.integers(0, 2, size=code_length).astype(int).tolist()


def _build_single_query(seed: int, bit: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(3, 32, 32, generator=generator)
    if bit == 1:
        query = query + 0.2
    return query


def build_positive_queries(codebook: list[int], seed: int) -> list[torch.Tensor]:
    return [
        _build_single_query(seed=seed + 97 * index, bit=bit)
        for index, bit in enumerate(codebook)
    ]


def build_negative_queries(codebook: list[int], seed: int) -> list[torch.Tensor]:
    flipped = [1 - bit for bit in codebook]
    return build_positive_queries(codebook=flipped, seed=seed + 104729)


def save_owner_artifacts(
    *,
    path: Path,
    owner_id: str,
    codebook: list[int],
    queries: list[torch.Tensor],
    negative_queries: list[torch.Tensor] | None = None,
    wm_train_config: dict[str, float] | None = None,
) -> None:
    if len(codebook) != len(queries):
        raise ValueError("codebook and queries must have the same length")
    if negative_queries is None:
        negative_queries = []
    if wm_train_config is None:
        wm_train_config = {}
    codebook_hash = hashlib.sha256(
        "".join(str(bit) for bit in codebook).encode("utf-8")
    ).hexdigest()
    payload = {
        "owner_id": owner_id,
        "codebook": codebook,
        "codebook_hash": codebook_hash,
        "positive_queries": [query.tolist() for query in queries],
        "negative_queries": [query.tolist() for query in negative_queries],
        "wm_train_config": wm_train_config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_owner_artifacts(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
