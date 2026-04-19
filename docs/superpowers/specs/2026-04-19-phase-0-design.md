# Phase 0 Scaffold Design

## Goal

Implement the documented Phase 0 repository scaffold for the research project: a minimal, installable, configuration-driven training smoke pipeline that is safe to extend in later phases.

## Scope

This phase only establishes repository structure and the smallest runnable loop.

Included:
- top-level repository directories required by the guide
- Python packaging and dependency definition
- a minimal Hydra configuration chain
- a tiny model builder in `src/models`
- a tiny FedLab-shaped smoke runner in `src/federated`
- a thin training entrypoint in `scripts/train`
- one smoke test in `tests/smoke`
- a README with setup and smoke-run instructions

Excluded:
- real dataset loading
- non-IID partitioning
- watermarking and verification
- attack pipelines
- evaluation and reporting logic beyond a minimal run record

## Design

### Repository layout

The repository will create the Phase 0 directories required by the guide:

```text
configs/
data/
docs/
outputs/
scripts/
src/
tests/
```

The `src/` package will include only the modules needed for the smoke loop:

```text
src/bbv/
  federated/
  models/
  utils/
```

### Configuration flow

Hydra will be the single configuration entrypoint. A top-level smoke config will describe:
- output root
- random seed
- synthetic federated setup
- minimal model dimensions
- optimizer settings

The thin script will load this config and pass it into the core runner. No experiment constants will live inside the script.

### Minimal federated smoke loop

The smoke loop will simulate a small number of clients over synthetic tensors. Each client will perform one local optimization step on a shared tiny classifier. The server side will average model parameters across clients and emit a run summary.

This keeps Phase 0 aligned with the guide's "minimal federated smoke test" intent while avoiding premature dataset and split logic.

### Outputs

Each smoke run will create a unique run directory under `outputs/runs/<run_id>/` containing a small JSON summary with configuration-derived metadata and final loss statistics. This satisfies the Phase 0 requirement to generate a basic run record without introducing full logging infrastructure.

### Testing

TDD will be used for the Phase 0 behavior:
- first add a smoke test that expects the pipeline to create a run summary
- run it and confirm failure
- implement the minimal scaffold
- rerun the smoke test and the smoke command

## Verification

Phase 0 verification commands:
- `pytest tests/smoke -q`
- `python scripts/train/run_smoke.py`

## Notes

This workspace is not currently a Git repository, so the design document cannot be committed yet. The file is still written in the location the skills expect so later work can reference it consistently.
