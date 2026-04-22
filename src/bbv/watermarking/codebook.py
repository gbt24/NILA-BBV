from __future__ import annotations

import hashlib

import numpy as np


def generate_codebook(owner_id: str, code_length: int, seed: int) -> list[int]:
    if code_length <= 0:
        raise ValueError("code_length must be greater than 0")
    owner_offset = sum(ord(ch) for ch in owner_id)
    rng = np.random.default_rng(seed + owner_offset)
    return rng.integers(0, 2, size=code_length).astype(int).tolist()


def compute_codebook_hash(codebook: list[int]) -> str:
    return hashlib.sha256("".join(str(bit) for bit in codebook).encode("utf-8")).hexdigest()
