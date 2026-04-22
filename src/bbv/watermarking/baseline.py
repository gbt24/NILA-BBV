"""Minimal watermark baseline artifacts for Phase 3."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from bbv.watermarking.codebook import compute_codebook_hash, generate_codebook
from bbv.watermarking.commitment import build_commitment_record
from bbv.watermarking.queries import build_negative_queries, build_positive_queries


def save_owner_artifacts(
    *,
    path: Path,
    owner_id: str,
    codebook: list[int],
    queries: list[torch.Tensor],
    negative_queries: list[torch.Tensor] | None = None,
    wm_train_config: dict[str, float] | None = None,
    seed: int | None = None,
) -> None:
    if len(codebook) != len(queries):
        raise ValueError("codebook and queries must have the same length")
    if negative_queries is None:
        negative_queries = []
    if wm_train_config is None:
        wm_train_config = {}
    payload = {
        "owner_id": owner_id,
        "codebook": codebook,
        "codebook_hash": compute_codebook_hash(codebook),
        "positive_queries": [query.tolist() for query in queries],
        "negative_queries": [query.tolist() for query in negative_queries],
        "wm_train_config": wm_train_config,
    }
    if seed is not None:
        payload["commitment"] = build_commitment_record(
            owner_id=owner_id,
            seed=seed,
            codebook=codebook,
            config=wm_train_config,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_owner_artifacts(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
