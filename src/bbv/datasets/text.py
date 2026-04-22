from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

TEXT_SEQUENCE_LENGTH = 32
TEXT_VOCAB_SIZE = 2048


def _normalize_sent140_label(raw_label: object) -> int:
    value = int(raw_label)
    return 0 if value <= 0 else 1


def _extract_text(raw_sample: object) -> str:
    if isinstance(raw_sample, str):
        return raw_sample
    if isinstance(raw_sample, list | tuple) and raw_sample:
        return str(raw_sample[-1])
    raise ValueError("expected Sent140 sample to be a string or sequence ending with text")


def _token_to_id(token: str) -> int:
    return (sum(ord(character) for character in token) % (TEXT_VOCAB_SIZE - 2)) + 2


def tokenize_text(text: str, *, sequence_length: int = TEXT_SEQUENCE_LENGTH) -> torch.Tensor:
    lowered = text.lower().strip()
    tokens = [token for token in lowered.split() if token]
    encoded = [_token_to_id(token) for token in tokens[:sequence_length]]
    if len(encoded) < sequence_length:
        encoded.extend([0] * (sequence_length - len(encoded)))
    return torch.tensor(encoded, dtype=torch.long)


class Sent140Dataset(Dataset):
    def __init__(self, samples: list[torch.Tensor], targets: list[int]) -> None:
        self.samples = samples
        self.targets = targets
        self.classes = ["negative", "positive"]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.samples[index], int(self.targets[index])


@dataclass(frozen=True)
class Sent140Split:
    dataset: Sent140Dataset
    client_indices: list[list[int]]
    user_ids: list[str]


def load_sent140_split(*, root: Path, train: bool) -> Sent140Split:
    split_name = "train" if train else "test"
    split_dir = root / "sent140" / split_name
    if not split_dir.exists():
        raise FileNotFoundError(
            "expected Sent140 split directory at "
            f"{split_dir} containing LEAF-style shard json files"
        )

    shard_paths = sorted(split_dir.glob("*.json"))
    if not shard_paths:
        raise FileNotFoundError(
            "expected Sent140 shard files under "
            f"{split_dir} (for example all_data_0.json)"
        )

    samples: list[torch.Tensor] = []
    targets: list[int] = []
    client_indices: list[list[int]] = []
    user_ids: list[str] = []

    for shard_path in shard_paths:
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
        user_data = payload.get("user_data", {})
        users = payload.get("users", list(user_data.keys()))
        for user_id in users:
            if user_id not in user_data:
                raise ValueError(f"missing user_data entry for {user_id} in {shard_path}")
            user_payload = user_data[user_id]
            raw_samples = user_payload.get("x", [])
            raw_labels = user_payload.get("y", [])
            if len(raw_samples) != len(raw_labels):
                raise ValueError(f"mismatched x/y lengths for {user_id} in {shard_path}")

            start = len(samples)
            for raw_sample, raw_label in zip(raw_samples, raw_labels, strict=True):
                samples.append(tokenize_text(_extract_text(raw_sample)))
                targets.append(_normalize_sent140_label(raw_label))
            stop = len(samples)
            if stop > start:
                client_indices.append(list(range(start, stop)))
                user_ids.append(str(user_id))

    if not samples:
        raise ValueError(f"no Sent140 samples found under {split_dir}")

    return Sent140Split(
        dataset=Sent140Dataset(samples=samples, targets=targets),
        client_indices=client_indices,
        user_ids=user_ids,
    )
