"""FPR evaluation: train N clean (non-owner) models and measure false-positive rate.

Usage:
    uv run python scripts/eval/run_fpr_evaluation.py \
        dataset=cifar10 \
        num_nonowners=20 \
        output_root=outputs/runs/fpr-nonowners \
        decision_threshold=0.5 \
        decision_margin=0.05
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.federated import train_federated
from bbv.verification.baseline import run_verification_from_checkpoint
from bbv.watermarking.codebook import generate_codebook
from bbv.watermarking.commitment import build_codebook_hash
from bbv.utils.io import write_json


@hydra.main(
    version_base=None,
    config_path="../../configs/eval",
    config_name="fpr",
)
def main(cfg: DictConfig) -> None:
    dataset = str(cfg.dataset.name)
    num_nonowners = int(cfg.get("num_nonowners", 20))
    output_root = Path(cfg.get("output_root", f"outputs/runs/{dataset}-fpr-nonowners"))
    tau = float(cfg.get("decision_threshold", 0.5))
    gamma = float(cfg.get("decision_margin", 0.05))
    competitor_ids = list(cfg.get("competitor_owner_ids", ["owner1", "owner2", "owner3", "owner4"]))
    code_length = int(cfg.watermarking.code_length)

    print(f"Training {num_nonowners} non-owner models on {dataset}...")
    print(f"FPR evaluation: tau={tau}, gamma={gamma}")

    for nonowner_seed in range(100, 100 + num_nonowners):
        train_federated(
            output_root=output_root,
            seed=nonowner_seed,
            dataset_name=dataset,
            model_name=str(cfg.model.name),
            num_classes=int(cfg.dataset.num_classes),
            num_clients=int(cfg.federated.num_clients),
            rounds=int(cfg.federated.rounds),
            participation_rate=float(cfg.federated.participation_rate),
            local_epochs=int(cfg.federated.local_epochs),
            batch_size=int(cfg.federated.batch_size),
            learning_rate=float(cfg.federated.learning_rate),
            samples_per_client=int(cfg.dataset.samples_per_client),
            partition_type=str(cfg.dataset.partition_type),
            concentration=float(cfg.dataset.concentration),
            shards_per_client=int(cfg.dataset.shards_per_client),
            quantity_sigma=float(cfg.dataset.quantity_sigma),
            watermark_hook=None,
            allocation_enabled=False,
            allocation_budget_ratio=0.0,
            allocation_base_loss_weight=0.0,
            progress_enabled=False,
            device=str(cfg.device),
        )

    print(f"\nRunning verification on all {num_nonowners} non-owner models...")
    owner_codebook = generate_codebook("owner0", code_length, 0)
    owner_codebook_hash = build_codebook_hash(owner_codebook)

    results = []
    for nonowner_seed in range(100, 100 + num_nonowners):
        run_dirs = sorted(
            output_root.glob(f"fedavg-*seed*{nonowner_seed}*"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not run_dirs:
            print(f"  [skip] seed {nonowner_seed}: no run dir found")
            continue
        run_dir = run_dirs[0]
        ckpt = run_dir / "best_checkpoint.pt"
        if not ckpt.exists():
            ckpt = run_dir / "checkpoint.pt"
        if not ckpt.exists():
            print(f"  [skip] seed {nonowner_seed}: no checkpoint")
            continue

        artifacts = run_dir / "owner_artifacts.json"
        if not artifacts.exists():
            print(f"  [skip] seed {nonowner_seed}: no artifacts")
            continue

        summary = run_verification_from_checkpoint(
            checkpoint_path=ckpt,
            artifacts_path=artifacts,
            decision_threshold=tau,
            margin=gamma,
            competitor_owner_ids=competitor_ids,
            seed=0,
            hard_label_only=True,
            device=str(cfg.device),
        )
        passed = summary["decision"]
        results.append({
            "seed": nonowner_seed,
            "owner_score": round(float(summary["owner_score"]), 4),
            "margin_value": round(float(summary["margin_value"]), 4),
            "passed": passed,
        })
        print(f"  seed {nonowner_seed}: score={results[-1]['owner_score']:.4f}, "
              f"margin={results[-1]['margin_value']:.4f}, passed={passed}")

    fp_count = sum(1 for r in results if r["passed"])
    fpr = fp_count / len(results) if results else float("nan")
    print(f"\n{'='*50}")
    print(f"FPR Summary: {fp_count}/{len(results)} passed")
    print(f"Empirical FPR (tau={tau}, gamma={gamma}): {fpr:.4f}")
    print(f"Owner codebook hash: {owner_codebook_hash}")

    threshold_sweep = []
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for g in [0.00, 0.03, 0.05, 0.07, 0.10]:
            fp = sum(1 for r in results if r["owner_score"] >= t and r["margin_value"] >= g)
            threshold_sweep.append({"tau": t, "gamma": g, "fp_count": fp, "total": len(results),
                                    "fpr": round(fp / len(results), 4) if results else float("nan")})
    print("\nThreshold sweep (tau, gamma → FPR):")
    for row in threshold_sweep:
        print(f"  tau={row['tau']:.2f}, gamma={row['gamma']:.2f}: "
              f"FPR={row['fpr']:.4f} ({row['fp_count']}/{row['total']})")

    report_path = output_root / "fpr_evaluation_report.json"
    write_json(report_path, {
        "num_nonowners": num_nonowners,
        "evaluated": len(results),
        "decision_threshold": tau,
        "decision_margin": gamma,
        "empirical_fpr": fpr,
        "per_seed_results": results,
        "threshold_sweep": threshold_sweep,
        "owner_codebook_hash": owner_codebook_hash,
    })
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
