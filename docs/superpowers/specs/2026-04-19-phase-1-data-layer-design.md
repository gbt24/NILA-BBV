# Phase 1 Data Layer Design

## Goal

Implement the documented Phase 1 repository slice: a real CIFAR-10 data loader, one reproducible Dirichlet non-IID partition method, and stable split metadata that later training stages can consume directly.

## Scope

This phase only builds the dataset and partition foundation.

Included:
- CIFAR-10 loading through a real dataset loader
- one reusable non-IID partition method based on Dirichlet label skew
- split metadata save and load logic under the standard data path
- unit tests for reproducibility, sample conservation, and metadata round-trip

Excluded:
- training-loop integration
- shard split or feature-skew split generation
- watermarking, verification, attacks, evaluation, or reporting
- synthetic dataset fallback for production code

## Design

### Repository layout

Phase 1 adds a focused dataset package under `src/bbv/`:

```text
src/bbv/
  datasets/
    __init__.py
    loaders.py
    partitions.py
    metadata.py
```

Tests stay isolated in `tests/unit/`, and persisted split artifacts live under `data/splits/`.

### Architecture

The data layer will be split into three small responsibilities:

1. `loaders.py`
Loads the real CIFAR-10 dataset and returns a small, explicit object containing the dataset instance plus lightweight metadata needed by downstream phases.

2. `partitions.py`
Builds a client partition from a sequence of labels using Dirichlet label skew only. The partitioner is deterministic under a fixed seed and returns per-client sample indices plus summary statistics.

3. `metadata.py`
Defines the stable split artifact schema and handles JSON serialization and deserialization. Phase 2 should be able to consume the saved metadata file without rebuilding the partition.

This keeps loading, split generation, and artifact persistence separate while avoiding a premature plugin or registry system.

### Interfaces

Expected Phase 1 interfaces:

- `load_dataset(root: Path, train: bool, download: bool) -> LoadedDataset`
- `build_partition(labels: Sequence[int], *, num_clients: int, concentration: float, seed: int) -> PartitionResult`
- `save_split_metadata(path: Path, metadata: SplitMetadata) -> None`
- `load_split_metadata(path: Path) -> SplitMetadata`

`LoadedDataset` should carry:
- dataset name
- train/test flag
- class count
- dataset size
- underlying dataset object

`PartitionResult` should carry:
- method name
- seed
- number of clients
- total sample count
- per-client sample indices
- per-client sample counts
- per-client label histograms

`SplitMetadata` should be the persisted JSON shape consumed by later phases. It should include:
- dataset name
- split method name: `dirichlet_label_skew`
- seed
- number of clients
- Dirichlet concentration
- total sample count
- per-client sample counts
- per-client sample indices
- per-client label histograms

### Data flow

Phase 1 data flow is:

1. Load CIFAR-10 from the configured root.
2. Extract labels from the training split.
3. Generate per-client indices with the Dirichlet partitioner under a fixed seed.
4. Save a canonical metadata artifact under `data/splits/...`.
5. Later phases reconstruct client subsets from the saved indices instead of recomputing the split.

This makes reproducibility depend on a saved artifact rather than hidden runtime state.

### Storage and artifact shape

The standard persistence target should be under `data/splits/` with a path shape that is stable and human-readable, for example:

```text
data/splits/cifar10/dirichlet_alpha-0.30_clients-10_seed-7.json
```

The JSON artifact should be self-describing enough for a later training stage to know:
- which dataset it belongs to
- how it was generated
- how many clients exist
- which indices belong to each client

No training-only fields should be added in this phase.

### Error handling

Phase 1 should enforce only the checks needed to keep artifacts trustworthy:

- reject `num_clients <= 0`
- reject non-positive Dirichlet concentration
- reject empty label input
- reject partitions that lose or duplicate samples
- fail loudly on malformed metadata instead of silently repairing it

This is enough to satisfy the phase exit conditions without introducing speculative validation layers.

### Testing

`tests/unit/test_splits.py` should cover:

- same seed produces identical partitions
- different seeds produce different partitions
- every sample appears exactly once across all clients
- total assigned sample count matches the source dataset
- metadata save and load round-trip without loss
- saved metadata contains the fields Phase 2 needs to reconstruct client subsets

The tests should focus on the partition logic and metadata artifact. They should not require full training or new integration infrastructure.

## Verification

Phase 1 verification commands:
- `uv run pytest tests/unit/test_splits.py -q`

Additional Phase 1 acceptance checks:
- repeat partition generation with the same seed and confirm identical output
- confirm assigned sample count equals original dataset sample count
- confirm saved split metadata can be loaded and consumed as a stable artifact

## Notes

This design intentionally stays with a single real dataset and a single partition method. Shard partitions, natural federated datasets, and richer split registries remain future extensions once the basic data artifact contract is stable.
