from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _reshape_femnist_image(raw_sample: object) -> torch.Tensor:
    tensor = torch.tensor(raw_sample, dtype=torch.float32)
    if tensor.numel() != 28 * 28:
        raise ValueError("FEMNIST samples must contain 784 values")
    image = tensor.reshape(1, 28, 28)
    padded = F.pad(image, (2, 2, 2, 2))
    return padded.repeat(3, 1, 1)


class FemnistDataset(Dataset):
    def __init__(self, samples: list[torch.Tensor], targets: list[int]) -> None:
        self.samples = samples
        self.targets = targets
        self.classes = [str(index) for index in range(62)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.samples[index], int(self.targets[index])


@dataclass(frozen=True)
class FemnistSplit:
    dataset: FemnistDataset
    client_indices: list[list[int]]
    user_ids: list[str]


def load_femnist_split(*, root: Path, train: bool) -> FemnistSplit:
    split_name = "train" if train else "test"
    split_dir = root / "femnist" / split_name
    if not split_dir.exists():
        raise FileNotFoundError(
            "expected FEMNIST split directory at "
            f"{split_dir} containing LEAF-style shard json files"
        )

    shard_paths = sorted(split_dir.glob("*.json"))
    if not shard_paths:
        raise FileNotFoundError(
            "expected FEMNIST shard files under "
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
                samples.append(_reshape_femnist_image(raw_sample))
                targets.append(int(raw_label))
            stop = len(samples)
            if stop > start:
                client_indices.append(list(range(start, stop)))
                user_ids.append(str(user_id))

    if not samples:
        raise ValueError(f"no FEMNIST samples found under {split_dir}")

    return FemnistSplit(
        dataset=FemnistDataset(samples=samples, targets=targets),
        client_indices=client_indices,
        user_ids=user_ids,
    )
