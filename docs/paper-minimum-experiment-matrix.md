# Paper Minimum Experiment Matrix (NILA-BBV)

This document contains runnable minimum experiment commands for generating paper-ready artifacts on CIFAR-10, CIFAR-100, and MNIST. Aligned with `deep-research-report-2.md` verification standards (FPR/FNR/ambiguity metrics, competitor margin, statistical calibration).

## 0) Environment

```bash
conda activate bbv
uv sync --extra dev
```

MNIST downloads automatically via torchvision on first use; no manual data preparation needed.

## 1) Main Experiments (3 seeds)

### 1.1 CIFAR-10 (adaptive)

```bash
for seed in 0 1 2; do
  uv run python scripts/train/run_watermark_baseline.py \
    dataset=cifar10 \
    allocation=adaptive \
    owner.id=owner0 \
    seed=$seed \
    federated.num_clients=50 \
    federated.rounds=200 \
    federated.participation_rate=0.2 \
    federated.local_epochs=5 \
    federated.batch_size=32 \
    federated.learning_rate=0.05 \
    dataset.samples_per_client=128 \
    allocation.budget_ratio=0.7 \
    watermarking.code_length=64 \
    watermarking.wm_weight=0.2 \
    output_root=outputs/runs/cifar10-main-adaptive
done
```

### 1.2 MNIST (adaptive)

```bash
for seed in 0 1 2; do
  uv run python scripts/train/run_watermark_baseline.py \
    dataset=mnist \
    allocation=adaptive \
    owner.id=owner0 \
    seed=$seed \
    federated.num_clients=50 \
    federated.rounds=80 \
    federated.participation_rate=0.2 \
    federated.local_epochs=3 \
    federated.batch_size=32 \
    federated.learning_rate=0.03 \
    dataset.samples_per_client=128 \
    allocation.budget_ratio=0.7 \
    watermarking.code_length=64 \
    watermarking.wm_weight=0.1 \
    output_root=outputs/runs/mnist-main-adaptive
done
```

### 1.3 CIFAR-100 (adaptive)

```bash
for seed in 0 1 2; do
  uv run python scripts/train/run_watermark_baseline.py \
    dataset=cifar100 \
    allocation=adaptive \
    owner.id=owner0 \
    seed=$seed \
    federated.num_clients=50 \
    federated.rounds=220 \
    federated.participation_rate=0.2 \
    federated.local_epochs=5 \
    federated.batch_size=32 \
    federated.learning_rate=0.05 \
    dataset.samples_per_client=128 \
    allocation.budget_ratio=0.7 \
    watermarking.code_length=64 \
    watermarking.wm_weight=0.1 \
    output_root=outputs/runs/cifar100-main-adaptive
done
```

## 2) Verification (margin + competitors, paper caliber)

### 2.1 CIFAR-10 verification

```bash
for seed in 0 1 2; do
  uv run python scripts/eval/run_verification.py \
    dataset=cifar10 \
    owner.id=owner0 \
    verification=margin \
    verification.competitor_owner_ids=[owner1,owner2,owner3,owner4] \
    verification.hard_label_only=true \
    verification.query_budget=64 \
    seed=$seed \
    output_root=outputs/runs/cifar10-main-adaptive
done
```

### 2.2 MNIST verification

```bash
for seed in 0 1 2; do
  uv run python scripts/eval/run_verification.py \
    dataset=mnist \
    owner.id=owner0 \
    verification=margin \
    verification.competitor_owner_ids=[owner1,owner2,owner3,owner4] \
    verification.hard_label_only=true \
    verification.query_budget=64 \
    seed=$seed \
    output_root=outputs/runs/mnist-main-adaptive
done
```

### 2.3 CIFAR-100 verification

```bash
for seed in 0 1 2; do
  uv run python scripts/eval/run_verification.py \
    dataset=cifar100 \
    owner.id=owner0 \
    verification=margin \
    verification.competitor_owner_ids=[owner1,owner2,owner3,owner4] \
    verification.hard_label_only=true \
    verification.query_budget=64 \
    seed=$seed \
    output_root=outputs/runs/cifar100-main-adaptive
done
```

## 3) Ablation (adaptive vs off)

### 3.1 CIFAR-10 ablation

```bash
for seed in 0 1 2; do
  uv run python scripts/train/run_watermark_baseline.py \
    dataset=cifar10 \
    allocation=off \
    owner.id=owner0 \
    seed=$seed \
    federated.num_clients=50 \
    federated.rounds=200 \
    federated.participation_rate=0.2 \
    federated.local_epochs=5 \
    federated.batch_size=32 \
    federated.learning_rate=0.05 \
    dataset.samples_per_client=128 \
    watermarking.code_length=64 \
    watermarking.wm_weight=0.1 \
    output_root=outputs/runs/cifar10-ablation-off
done
```

### 3.2 MNIST ablation

```bash
for seed in 0 1 2; do
  uv run python scripts/train/run_watermark_baseline.py \
    dataset=mnist \
    allocation=off \
    owner.id=owner0 \
    seed=$seed \
    federated.num_clients=50 \
    federated.rounds=80 \
    federated.participation_rate=0.2 \
    federated.local_epochs=3 \
    federated.batch_size=32 \
    federated.learning_rate=0.03 \
    dataset.samples_per_client=128 \
    watermarking.code_length=64 \
    watermarking.wm_weight=0.1 \
    output_root=outputs/runs/mnist-ablation-off
done
```

### 3.3 Ablation verification

```bash
for ds in cifar10 mnist; do
  for seed in 0 1 2; do
    uv run python scripts/eval/run_verification.py \
      dataset=$ds \
      owner.id=owner0 \
      verification=margin \
      verification.competitor_owner_ids=[owner1,owner2] \
      verification.hard_label_only=true \
      verification.query_budget=64 \
      seed=$seed \
      output_root=outputs/runs/${ds}-ablation-off
  done
done
```

## 4) Non-IID Intensity Sweep (Fig.4)

### 4.1 Dirichlet alpha sweep (CIFAR-10)

```bash
for alpha in 0.1 0.3 0.5 1.0; do
  for seed in 0 1 2; do
    uv run python scripts/train/run_watermark_baseline.py \
      dataset=cifar10 \
      dataset.partition_type=dirichlet \
      dataset.concentration=$alpha \
      allocation=adaptive \
      owner.id=owner0 \
      seed=$seed \
      federated.num_clients=50 \
      federated.rounds=120 \
      federated.participation_rate=0.2 \
      federated.local_epochs=3 \
      federated.batch_size=32 \
      dataset.samples_per_client=128 \
      watermarking.code_length=64 \
      watermarking.wm_weight=0.2 \
      output_root=outputs/runs/cifar10-dirichlet-a${alpha}
  done
done
```

```bash
for alpha in 0.1 0.3 0.5 1.0; do
  for seed in 0 1 2; do
    uv run python scripts/eval/run_verification.py \
      dataset=cifar10 \
      owner.id=owner0 \
      verification=margin \
      verification.competitor_owner_ids=[owner1,owner2] \
      verification.hard_label_only=true \
      verification.query_budget=64 \
      seed=$seed \
      output_root=outputs/runs/cifar10-dirichlet-a${alpha}
  done
done
```

### 4.2 Dirichlet alpha sweep (MNIST)

```bash
for alpha in 0.1 0.3 0.5 1.0; do
  for seed in 0 1 2; do
    uv run python scripts/train/run_watermark_baseline.py \
      dataset=mnist \
      dataset.partition_type=dirichlet \
      dataset.concentration=$alpha \
      allocation=adaptive \
      owner.id=owner0 \
      seed=$seed \
      federated.num_clients=50 \
      federated.rounds=60 \
      federated.participation_rate=0.2 \
      federated.local_epochs=2 \
      federated.batch_size=32 \
      dataset.samples_per_client=128 \
      watermarking.code_length=64 \
      watermarking.wm_weight=0.1 \
      output_root=outputs/runs/mnist-dirichlet-a${alpha}
  done
done
```

```bash
for alpha in 0.1 0.3 0.5 1.0; do
  for seed in 0 1 2; do
    uv run python scripts/eval/run_verification.py \
      dataset=mnist \
      owner.id=owner0 \
      verification=margin \
      verification.competitor_owner_ids=[owner1,owner2] \
      verification.hard_label_only=true \
      verification.query_budget=64 \
      seed=$seed \
      output_root=outputs/runs/mnist-dirichlet-a${alpha}
  done
done
```

### 4.3 Quantity skew sweep (CIFAR-10)

```bash
for sigma in 0.5 1.0; do
  for seed in 0 1 2; do
    uv run python scripts/train/run_watermark_baseline.py \
      dataset=cifar10 \
      dataset.partition_type=quantity_skew \
      dataset.quantity_sigma=$sigma \
      allocation=adaptive \
      owner.id=owner0 \
      seed=$seed \
      federated.num_clients=50 \
      federated.rounds=120 \
      federated.participation_rate=0.2 \
      federated.local_epochs=3 \
      federated.batch_size=32 \
      dataset.samples_per_client=128 \
      watermarking.code_length=64 \
      watermarking.wm_weight=0.2 \
      output_root=outputs/runs/cifar10-quantity-s${sigma}
  done
done
```

```bash
for sigma in 0.5 1.0; do
  for seed in 0 1 2; do
    uv run python scripts/eval/run_verification.py \
      dataset=cifar10 \
      owner.id=owner0 \
      verification=margin \
      verification.competitor_owner_ids=[owner1,owner2] \
      verification.hard_label_only=true \
      verification.query_budget=64 \
      seed=$seed \
      output_root=outputs/runs/cifar10-quantity-s${sigma}
  done
done
```

## 5) Robustness (5 attacks)

For each dataset/main-run, set `RUN_DIR` to the corresponding run directory and execute all attack types.

### 5.1 CIFAR-10 robustness

```bash
# Example for seed0: replace <run-id> with actual directory name
RUN_DIR=outputs/runs/cifar10-main-adaptive/<run-id-for-seed0>
OUT=outputs/attacks/cifar10-robustness-seed0

uv run python scripts/attacks/run_attack_suite.py attack=finetune    dataset=cifar10 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=pruning     dataset=cifar10 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=quantization dataset=cifar10 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=distillation dataset=cifar10 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=extraction  dataset=cifar10 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
```

### 5.2 MNIST robustness

```bash
# Example for seed0: replace <run-id> with actual directory name
RUN_DIR=outputs/runs/mnist-main-adaptive/<run-id-for-seed0>
OUT=outputs/attacks/mnist-robustness-seed0

uv run python scripts/attacks/run_attack_suite.py attack=finetune    dataset=mnist checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=pruning     dataset=mnist checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=quantization dataset=mnist checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=distillation dataset=mnist checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=extraction  dataset=mnist checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
```

### 5.3 CIFAR-100 robustness

```bash
# Example for seed0: replace <run-id> with actual directory name
RUN_DIR=outputs/runs/cifar100-main-adaptive/<run-id-for-seed0>
OUT=outputs/attacks/cifar100-robustness-seed0

uv run python scripts/attacks/run_attack_suite.py attack=finetune    dataset=cifar100 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=pruning     dataset=cifar100 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=quantization dataset=cifar100 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=distillation dataset=cifar100 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
uv run python scripts/attacks/run_attack_suite.py attack=extraction  dataset=cifar100 checkpoint=$RUN_DIR/best_checkpoint.pt seed=0 output_root=$OUT
```

## 6) Report Export (tables/figures/summaries)

### 6.1 CIFAR-10

```bash
uv run python scripts/report/build_report.py dataset=cifar10 study=main outputs_dir=outputs/runs/cifar10-main-adaptive attacks_dir=outputs/attacks report_root=outputs/cifar10-main-report
uv run python scripts/report/build_report.py dataset=cifar10 study=main outputs_dir=outputs/runs/cifar10-ablation-off attacks_dir=outputs/attacks report_root=outputs/cifar10-ablation-report
uv run python scripts/report/build_report.py dataset=cifar10 study=main outputs_dir=outputs/runs/cifar10-main-adaptive attacks_dir=outputs/attacks report_root=outputs/cifar10-robustness-report
```

### 6.2 MNIST

```bash
uv run python scripts/report/build_report.py dataset=mnist study=main outputs_dir=outputs/runs/mnist-main-adaptive attacks_dir=outputs/attacks report_root=outputs/mnist-main-report
uv run python scripts/report/build_report.py dataset=mnist study=main outputs_dir=outputs/runs/mnist-ablation-off attacks_dir=outputs/attacks report_root=outputs/mnist-ablation-report
uv run python scripts/report/build_report.py dataset=mnist study=main outputs_dir=outputs/runs/mnist-main-adaptive attacks_dir=outputs/attacks report_root=outputs/mnist-robustness-report
```

### 6.3 CIFAR-100

```bash
uv run python scripts/report/build_report.py dataset=cifar100 study=main outputs_dir=outputs/runs/cifar100-main-adaptive attacks_dir=outputs/attacks report_root=outputs/cifar100-main-report
uv run python scripts/report/build_report.py dataset=cifar100 study=main outputs_dir=outputs/runs/cifar100-main-adaptive attacks_dir=outputs/attacks report_root=outputs/cifar100-robustness-report
```

## 7) Final Artifacts to Collect

- `outputs/tables/*-main-results.csv`
- `outputs/tables/*-ablation-results.csv`
- `outputs/tables/*-robustness-results.csv`
- `outputs/tables/attack-robustness.csv`
- `outputs/figures/owner-nonowner-score-distribution.svg`
- `outputs/figures/*-tradeoff-figure.svg`
- `outputs/summaries/*-summary.md`

Raw evidence:

- `outputs/runs/**/verification_margin_summary.json`
- `outputs/runs/**/calibration_artifacts.json`
- `outputs/attacks/**/verification_after_attack.json`
