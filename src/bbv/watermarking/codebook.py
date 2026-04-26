from __future__ import annotations

import hashlib

import numpy as np


def generate_codebook(owner_id: str, code_length: int, seed: int) -> list[int]:
    if code_length <= 0:
        raise ValueError("code_length must be greater than 0")
    owner_offset = sum(ord(ch) for ch in owner_id)
    rng = np.random.default_rng(seed + owner_offset)
    return rng.integers(0, 2, size=code_length).astype(int).tolist()


def generate_single_trigger_codebook(code_length: int) -> list[int]:
    """Single-trigger baseline codebook: all bits = 1 (single trigger pattern)."""
    return [1] * code_length


def generate_hadamard_codebook(owner_index: int, code_length: int) -> list[int]:
    """Hadamard codebook: maximally separated rows of a Hadamard matrix.

    Requires code_length be a power of 2.
    Row 0 (all-ones) is skipped; uses row (owner_index + 1) % code_length.
    """
    if code_length & (code_length - 1) != 0:
        raise ValueError("code_length must be a power of 2 for Hadamard codes")
    H = np.zeros((code_length, code_length), dtype=int)
    H[0, :] = 1
    for k in range(1, code_length):
        H[k] = H[k - 1].copy()
        H[k, k - 1] = -H[k, k - 1]
        for j in range(k + 1, code_length):
            H[k, j] = 1
    row_idx = (owner_index + 1) % code_length
    return [int(H[row_idx, b] > 0) for b in range(code_length)]


def generate_maximally_separated_codebooks(
    code_length: int, num_owners: int, seed: int
) -> list[list[int]]:
    """Greedy selection of codebooks with maximal pairwise Hamming distance."""
    rng = np.random.default_rng(seed)
    candidates = [rng.integers(0, 2, size=code_length).tolist()
                  for _ in range(num_owners * 20)]
    selected = [candidates[0]]
    for _ in range(num_owners - 1):
        best = max(candidates, key=lambda c: min(
            sum(a != b for a, b in zip(c, s)) for s in selected
        ))
        selected.append(best)
    return selected


def compute_codebook_hash(codebook: list[int]) -> str:
    return hashlib.sha256("".join(str(bit) for bit in codebook).encode("utf-8")).hexdigest()
