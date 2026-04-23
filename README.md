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

## Phase 2 FedAvg Baseline Verification

Run the minimal FedAvg baseline command:

```bash
uv run python scripts/train/run_fedavg.py dataset=cifar10 federated=fedavg model=resnet18 seed=0
```

A successful run writes outputs under:

```text
outputs/runs/fedavg-*/
```

Expected files in each run directory:

- `metrics.json`
- `run_metadata.json`
- `checkpoint.pt`

## Phase 3 Watermark Baseline Verification

Run the Phase 3 baseline command:

```bash
uv run python scripts/train/run_watermark_baseline.py dataset=cifar10 watermarking=baseline owner.id=owner0 seed=0
```

A successful run writes additional artifacts under the generated `outputs/runs/fedavg-*/` directory:

- `owner_artifacts.json`
- `verification_summary.json`

## Phase 4 Adaptive Allocation Verification

Run the allocation integration test:

```bash
uv run pytest tests/integration/test_allocation_pipeline.py -q
```

Run watermark baseline with adaptive allocation enabled:

```bash
uv run python scripts/train/run_watermark_baseline.py dataset=cifar10 watermarking=baseline allocation=adaptive owner.id=owner0 seed=0
```

When enabled, each run directory additionally includes:

- `allocation_assignments.json`

## Phase 5 Verification and Calibration

Run the Phase 5 verification command:

```bash
uv run python scripts/eval/run_verification.py dataset=cifar10 verification=margin owner.id=owner0 seed=0
```

The selected run directory includes:

- `verification_margin_summary.json`
- `calibration_artifacts.json`

## Phase 6 Attack Suite Verification

Run the Phase 6 attack suite command:

```bash
uv run python scripts/attacks/run_attack_suite.py attack=finetune dataset=cifar10 checkpoint=outputs/runs/fedavg-20260419-144452-b8dca20a/checkpoint.pt seed=0
```

Each attack run writes outputs under `outputs/attacks/<attack-run-id>/`:

- `attacked_checkpoint.pt`
- `attack_log.json`

## Phase 7 Report Export

Run the Phase 7 report command:

```bash
uv run python scripts/report/build_report.py dataset=cifar10 study=main outputs_dir=outputs/runs/cifar10-main-seed0
```

Generated artifacts are exported to:

- `outputs/figures/`
- `outputs/tables/`
- `outputs/summaries/`

## Research-Grade Main Experiment Matrix

The repository now exposes a standard 3-seed experiment surface through the main configs:

- `configs/train/main.yaml`
- `configs/eval/main.yaml`
- `configs/attacks/main.yaml`
- `configs/report/main.yaml`

Default matrix metadata:

- `seeds: [0, 1, 2]`
- training/report studies: `main`, `ablation`, `false_claim`, `robustness`
- attack studies: `robustness`

Supported primary datasets for the first research matrix:

- `cifar10`
- `cifar100`
- `femnist`
- `sent140`

Recommended commands:

```bash
uv run python scripts/train/run_watermark_baseline.py dataset=cifar10 allocation=adaptive owner.id=owner0 seed=0
uv run python scripts/train/run_watermark_baseline.py dataset=cifar10 allocation=adaptive owner.id=owner0 seed=1
uv run python scripts/train/run_watermark_baseline.py dataset=cifar10 allocation=adaptive owner.id=owner0 seed=2

uv run python scripts/eval/run_verification.py dataset=cifar10 verification=margin owner.id=owner0 seed=0

uv run python scripts/attacks/run_attack_suite.py attack=finetune dataset=cifar10
uv run python scripts/attacks/run_attack_suite.py attack=distillation dataset=cifar10
uv run python scripts/attacks/run_attack_suite.py attack=extraction dataset=cifar10

uv run python scripts/report/build_report.py dataset=cifar10 study=main outputs_dir=outputs/runs attacks_dir=outputs/attacks
```

Dataset-specific config checks:

```bash
uv run python scripts/train/run_watermark_baseline.py --cfg job dataset=cifar100 allocation=adaptive
uv run python scripts/eval/run_verification.py --cfg job dataset=cifar100 verification=margin
uv run python scripts/report/build_report.py --cfg job dataset=cifar100 study=main
```

## Preparing LEAF Datasets

To automatically download and preprocess the LEAF-style datasets used by this repository:

```bash
uv run python scripts/data/prepare_leaf_datasets.py --dataset=femnist
uv run python scripts/data/prepare_leaf_datasets.py --dataset=sent140
```

Or prepare both:

```bash
uv run python scripts/data/prepare_leaf_datasets.py --dataset=all
```

Default output layout:

- `data/raw/femnist/train/*.json`
- `data/raw/femnist/test/*.json`
- `data/raw/sent140/train/*.json`
- `data/raw/sent140/test/*.json`
