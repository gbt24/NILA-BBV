# BBV

Phase 0 scaffold for the research codebase on non-IID-aware, low-ambiguity, adaptive black-box copyright verification in federated learning.

## Phase 0 Smoke Run

Create the environment and install dependencies:

```bash
uv sync --extra dev
```

Run the smoke test:

```bash
uv run pytest tests/smoke -q
```

Run the minimal training smoke command:

```bash
uv run python scripts/train/run_smoke.py
```

## Phase 1 Data Layer Verification

Run the split unit tests:

```bash
uv run pytest tests/unit/test_splits.py -q
```

Phase 1 split artifacts are expected under paths like:

```text
data/splits/cifar10/dirichlet_alpha-0.30_clients-10_seed-7.json
```
