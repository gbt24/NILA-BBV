# Phase 0 Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 0 repository scaffold with a configuration-driven smoke training loop and a smoke test that proves a run record is generated.

**Architecture:** Keep `scripts/` thin and place all runnable logic in `src/bbv`. Use Hydra for configuration, a tiny synthetic federated loop for the smoke pipeline, and a JSON run summary in `outputs/runs` as the only persisted artifact.

**Tech Stack:** Python 3.11, PyTorch, Hydra, pytest

---

### Task 1: Add the failing smoke test and repository skeleton files

**Files:**
- Create: `tests/smoke/test_smoke_run.py`
- Create: `src/bbv/__init__.py`
- Create: `src/bbv/federated/__init__.py`
- Create: `src/bbv/models/__init__.py`
- Create: `src/bbv/utils/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from bbv.federated.runner import run_smoke_experiment


def test_smoke_run_writes_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    result = run_smoke_experiment(output_root=output_dir)

    assert result.run_dir.exists()
    assert result.summary_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_smoke_run.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing `run_smoke_experiment`

- [ ] **Step 3: Create package skeleton files**

```python
# __init__.py files can be empty or contain a short package marker.
```

- [ ] **Step 4: Re-run the failing test**

Run: `pytest tests/smoke/test_smoke_run.py -q`
Expected: still FAIL because the runner is not implemented yet

### Task 2: Add packaging, config, runner, and thin entrypoint

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `configs/train/smoke.yaml`
- Create: `src/bbv/models/simple.py`
- Create: `src/bbv/federated/runner.py`
- Create: `src/bbv/utils/io.py`
- Create: `scripts/train/run_smoke.py`

- [ ] **Step 1: Write minimal implementation for the model builder**

```python
import torch.nn as nn


def build_simple_classifier(input_dim: int, num_classes: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, num_classes))
```

- [ ] **Step 2: Write minimal implementation for the run-summary helper**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    summary_path: Path
```

- [ ] **Step 3: Write the minimal smoke runner**

```python
def run_smoke_experiment(output_root: Path, seed: int = 7) -> RunResult:
    ...
```

Behavior:
- build synthetic client tensors
- run one local update per client
- average parameters
- create `outputs/runs/<run_id>/summary.json`

- [ ] **Step 4: Add Hydra config and thin script**

Run script behavior:
- load `configs/train/smoke.yaml`
- call `run_smoke_experiment(...)`
- print the summary path

- [ ] **Step 5: Run smoke test to verify it passes**

Run: `pytest tests/smoke/test_smoke_run.py -q`
Expected: PASS

### Task 3: Verify the documented Phase 0 exit path

**Files:**
- Update: `tests/smoke/test_smoke_run.py`
- Update: `README.md`

- [ ] **Step 1: Expand the smoke test assertions slightly**

```python
assert result.metrics["num_clients"] > 0
assert result.metrics["final_loss"] >= 0.0
```

- [ ] **Step 2: Run the smoke test suite**

Run: `pytest tests/smoke -q`
Expected: PASS

- [ ] **Step 3: Run the training smoke command**

Run: `python scripts/train/run_smoke.py`
Expected: exits 0 and prints a summary path under `outputs/runs/`

- [ ] **Step 4: Verify a run record exists**

Run: inspect `outputs/runs/` and confirm a new run directory contains `summary.json`
