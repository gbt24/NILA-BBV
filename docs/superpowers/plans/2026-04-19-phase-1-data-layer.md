# Phase 1 Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 data layer with a real CIFAR-10 loader, a reproducible Dirichlet non-IID partitioner, and stable split metadata that later training code can consume directly.

**Architecture:** Add a small `src/bbv/datasets` package with three focused modules: dataset loading, partition generation, and metadata persistence. Keep the partition contract explicit and JSON-serializable so Phase 2 can reconstruct client subsets from saved indices instead of regenerating splits.

**Tech Stack:** Python 3.11, PyTorch, torchvision, pytest

---

### Task 1: Add the failing unit tests and dataset package skeleton

**Files:**
- Create: `src/bbv/datasets/__init__.py`
- Create: `tests/unit/test_splits.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from bbv.datasets.metadata import SplitMetadata, load_split_metadata, save_split_metadata
from bbv.datasets.partitions import build_partition


def test_dirichlet_partition_is_reproducible() -> None:
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    left = build_partition(labels, num_clients=3, concentration=0.5, seed=7)
    right = build_partition(labels, num_clients=3, concentration=0.5, seed=7)

    assert left.client_indices == right.client_indices
    assert left.client_label_histograms == right.client_label_histograms


def test_dirichlet_partition_conserves_samples() -> None:
    labels = [0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2]

    result = build_partition(labels, num_clients=4, concentration=0.3, seed=11)
    flattened = sorted(index for indices in result.client_indices for index in indices)

    assert flattened == list(range(len(labels)))
    assert sum(result.client_sample_counts) == len(labels)


def test_split_metadata_round_trip(tmp_path: Path) -> None:
    metadata = SplitMetadata(
        dataset_name="cifar10",
        split_method="dirichlet_label_skew",
        seed=7,
        num_clients=3,
        concentration=0.5,
        total_samples=6,
        client_sample_counts=[2, 2, 2],
        client_indices=[[0, 3], [1, 4], [2, 5]],
        client_label_histograms=[
            {"0": 1, "1": 1},
            {"0": 1, "1": 1},
            {"0": 1, "1": 1},
        ],
    )

    path = tmp_path / "split.json"
    save_split_metadata(path, metadata)

    loaded = load_split_metadata(path)

    assert loaded == metadata


def test_dirichlet_partition_rejects_invalid_client_count() -> None:
    with pytest.raises(ValueError, match="num_clients"):
        build_partition([0, 1, 0], num_clients=0, concentration=0.5, seed=7)
```

- [ ] **Step 2: Run the unit test file to verify it fails**

Run: `uv run pytest tests/unit/test_splits.py -q`
Expected: FAIL with `ModuleNotFoundError` for `bbv.datasets` or missing symbols

- [ ] **Step 3: Create the dataset package marker**

```python
"""Dataset loading and partition helpers."""
```

- [ ] **Step 4: Re-run the unit test file**

Run: `uv run pytest tests/unit/test_splits.py -q`
Expected: still FAIL because the dataset modules are not implemented yet

### Task 2: Add metadata persistence and make the round-trip test pass

**Files:**
- Create: `src/bbv/datasets/metadata.py`
- Modify: `tests/unit/test_splits.py`

- [ ] **Step 1: Implement the metadata model and JSON persistence**

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitMetadata:
    dataset_name: str
    split_method: str
    seed: int
    num_clients: int
    concentration: float
    total_samples: int
    client_sample_counts: list[int]
    client_indices: list[list[int]]
    client_label_histograms: list[dict[str, int]]


def save_split_metadata(path: Path, metadata: SplitMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_split_metadata(path: Path) -> SplitMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitMetadata(**payload)
```

- [ ] **Step 2: Add one malformed-metadata test**

```python
def test_split_metadata_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"dataset_name": "cifar10"}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        load_split_metadata(path)
```

- [ ] **Step 3: Run only the metadata tests**

Run: `uv run pytest tests/unit/test_splits.py -q -k metadata`
Expected: PASS for metadata round-trip and malformed-file tests, while partition tests still FAIL

- [ ] **Step 4: Export the metadata API from the package**

```python
from bbv.datasets.metadata import SplitMetadata, load_split_metadata, save_split_metadata

__all__ = ["SplitMetadata", "load_split_metadata", "save_split_metadata"]
```

### Task 3: Implement the Dirichlet partitioner and make the partition tests pass

**Files:**
- Create: `src/bbv/datasets/partitions.py`
- Modify: `src/bbv/datasets/__init__.py`
- Modify: `tests/unit/test_splits.py`

- [ ] **Step 1: Implement the partition result model and validation helpers**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PartitionResult:
    method_name: str
    seed: int
    num_clients: int
    total_samples: int
    client_indices: list[list[int]]
    client_sample_counts: list[int]
    client_label_histograms: list[dict[str, int]]


def _validate_partition_inputs(
    labels: Sequence[int], num_clients: int, concentration: float
) -> None:
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0")
    if concentration <= 0:
        raise ValueError("concentration must be greater than 0")
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
```

- [ ] **Step 2: Implement the Dirichlet label-skew partitioner**

```python
def build_partition(
    labels: Sequence[int], *, num_clients: int, concentration: float, seed: int
) -> PartitionResult:
    _validate_partition_inputs(labels, num_clients, concentration)
    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels, dtype=np.int64)
    client_buckets: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in np.unique(label_array):
        class_indices = np.flatnonzero(label_array == class_id)
        shuffled = rng.permutation(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, concentration))
        cut_points = np.cumsum(proportions[:-1]) * len(shuffled)
        splits = np.split(shuffled, cut_points.astype(int))
        for client_index, split in enumerate(splits):
            client_buckets[client_index].extend(int(index) for index in split)

    normalized_indices = [sorted(indices) for indices in client_buckets]
    flattened = sorted(index for indices in normalized_indices for index in indices)
    expected = list(range(len(labels)))
    if flattened != expected:
        raise ValueError("partition must cover every sample exactly once")

    histograms: list[dict[str, int]] = []
    for indices in normalized_indices:
        counts: dict[str, int] = {}
        for index in indices:
            label = str(int(label_array[index]))
            counts[label] = counts.get(label, 0) + 1
        histograms.append(counts)

    return PartitionResult(
        method_name="dirichlet_label_skew",
        seed=seed,
        num_clients=num_clients,
        total_samples=len(labels),
        client_indices=normalized_indices,
        client_sample_counts=[len(indices) for indices in normalized_indices],
        client_label_histograms=histograms,
    )
```

- [ ] **Step 3: Add one different-seed test to prove the seed matters**

```python
def test_dirichlet_partition_changes_with_seed() -> None:
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    left = build_partition(labels, num_clients=3, concentration=0.5, seed=7)
    right = build_partition(labels, num_clients=3, concentration=0.5, seed=8)

    assert left.client_indices != right.client_indices
```

- [ ] **Step 4: Export the partition API from the package**

```python
from bbv.datasets.metadata import SplitMetadata, load_split_metadata, save_split_metadata
from bbv.datasets.partitions import PartitionResult, build_partition

__all__ = [
    "PartitionResult",
    "SplitMetadata",
    "build_partition",
    "load_split_metadata",
    "save_split_metadata",
]
```

- [ ] **Step 5: Run the partition-focused tests**

Run: `uv run pytest tests/unit/test_splits.py -q -k partition`
Expected: PASS for reproducibility, different-seed, conservation, and invalid-input tests

### Task 4: Add the real CIFAR-10 loader and stable split artifact builder

**Files:**
- Create: `src/bbv/datasets/loaders.py`
- Modify: `src/bbv/datasets/metadata.py`
- Modify: `src/bbv/datasets/__init__.py`
- Modify: `tests/unit/test_splits.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `torchvision` to runtime dependencies**

```toml
dependencies = [
  "hydra-core>=1.3,<2.0",
  "numpy>=1.26,<3.0",
  "torch>=2.2,<3.0",
  "torchvision>=0.17,<1.0",
]
```

- [ ] **Step 2: Implement the real CIFAR-10 loader**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torchvision import datasets


@dataclass(frozen=True)
class LoadedDataset:
    dataset_name: str
    train: bool
    num_classes: int
    num_samples: int
    dataset: datasets.CIFAR10


def load_dataset(root: Path, train: bool, download: bool) -> LoadedDataset:
    dataset = datasets.CIFAR10(root=str(root), train=train, download=download)
    return LoadedDataset(
        dataset_name="cifar10",
        train=train,
        num_classes=len(dataset.classes),
        num_samples=len(dataset),
        dataset=dataset,
    )
```

- [ ] **Step 3: Add a helper that converts a partition result into stable metadata**

```python
from bbv.datasets.metadata import SplitMetadata
from bbv.datasets.partitions import PartitionResult


def build_split_metadata(
    dataset_name: str, concentration: float, partition: PartitionResult
) -> SplitMetadata:
    return SplitMetadata(
        dataset_name=dataset_name,
        split_method=partition.method_name,
        seed=partition.seed,
        num_clients=partition.num_clients,
        concentration=concentration,
        total_samples=partition.total_samples,
        client_sample_counts=partition.client_sample_counts,
        client_indices=partition.client_indices,
        client_label_histograms=partition.client_label_histograms,
    )
```

- [ ] **Step 4: Add a test that Phase 2 can reconstruct client subsets from saved indices**

```python
def test_saved_metadata_can_drive_subset_reconstruction(tmp_path: Path) -> None:
    labels = [0, 1, 0, 1, 2, 2]
    partition = build_partition(labels, num_clients=3, concentration=0.5, seed=7)
    metadata = build_split_metadata("cifar10", 0.5, partition)
    path = tmp_path / "cifar10" / "split.json"
    save_split_metadata(path, metadata)

    loaded = load_split_metadata(path)
    reconstructed = [labels[index] for index in loaded.client_indices[0]]

    assert len(loaded.client_indices) == 3
    assert len(reconstructed) == loaded.client_sample_counts[0]
```

- [ ] **Step 5: Export the loader API from the package**

```python
from bbv.datasets.loaders import LoadedDataset, load_dataset
from bbv.datasets.metadata import (
    SplitMetadata,
    build_split_metadata,
    load_split_metadata,
    save_split_metadata,
)
from bbv.datasets.partitions import PartitionResult, build_partition

__all__ = [
    "LoadedDataset",
    "PartitionResult",
    "SplitMetadata",
    "build_partition",
    "build_split_metadata",
    "load_dataset",
    "load_split_metadata",
    "save_split_metadata",
]
```

- [ ] **Step 6: Sync dependencies and run the full unit test file**

Run: `uv sync --extra dev && uv run pytest tests/unit/test_splits.py -q`
Expected: PASS

### Task 5: Document the split artifact path and verify the Phase 1 exit condition

**Files:**
- Modify: `README.md`
- Modify: `tests/unit/test_splits.py`

- [ ] **Step 1: Add one README section for the Phase 1 verification target**

```md
## Phase 1 Data Layer Verification

Run the split unit tests:

    uv run pytest tests/unit/test_splits.py -q

Phase 1 split artifacts are expected under paths like:

    data/splits/cifar10/dirichlet_alpha-0.30_clients-10_seed-7.json
```

- [ ] **Step 2: Add one test for a canonical save path layout**

```python
def test_split_metadata_is_saved_under_expected_path(tmp_path: Path) -> None:
    partition = build_partition([0, 1, 0, 1], num_clients=2, concentration=0.3, seed=7)
    metadata = build_split_metadata("cifar10", 0.3, partition)
    path = tmp_path / "data" / "splits" / "cifar10" / "dirichlet_alpha-0.30_clients-2_seed-7.json"

    save_split_metadata(path, metadata)

    assert path.exists()
```

- [ ] **Step 3: Run the documented verification command**

Run: `uv run pytest tests/unit/test_splits.py -q`
Expected: PASS

- [ ] **Step 4: Run one direct reproducibility check in the interpreter**

Run: `uv run python -c "from bbv.datasets.partitions import build_partition; labels=[0,0,0,1,1,1,2,2,2,2]; a=build_partition(labels, num_clients=3, concentration=0.5, seed=7); b=build_partition(labels, num_clients=3, concentration=0.5, seed=7); print(a.client_indices == b.client_indices)"`
Expected: prints `True`

- [ ] **Step 5: Commit**

```bash
git add README.md pyproject.toml src/bbv/datasets tests/unit/test_splits.py docs/superpowers/specs/2026-04-19-phase-1-data-layer-design.md docs/superpowers/plans/2026-04-19-phase-1-data-layer.md
git commit -m "feat: add phase 1 data layer and split artifacts"
```
