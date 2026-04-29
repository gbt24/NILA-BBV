"""FPR evaluation: train N clean (non-owner) models and measure false-positive rate.

Usage:
    # Combined train + verify (fresh run):
    uv run python scripts/eval/run_fpr_evaluation.py \
        dataset=cifar10 num_nonowners=20 output_root=outputs/runs/fpr-nonowners

    # Verify-only mode (skip training, read existing models by metadata):
    uv run python scripts/eval/run_fpr_evaluation.py \
        output_root=outputs/runs/cifar10-fpr-nonowners \
        num_nonowners=20 skip_training=True dataset.name=cifar10
"""

from __future__ import annotations

import json as _json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.federated import train_federated
from bbv.verification.baseline import run_verification_from_checkpoint
from bbv.watermarking.codebook import (
    generate_codebook,
    generate_hadamard_codebook,
    generate_single_trigger_codebook,
)
from bbv.watermarking.commitment import build_codebook_hash
from bbv.watermarking.queries import build_negative_queries, build_positive_queries
from bbv.utils.io import write_json


def _find_run_dir_by_seed(output_root: Path, target_seed: int) -> Path | None:
    """Find run directory by reading run_metadata.json of each fedavg-* subdir."""
    candidates = sorted(output_root.glob("fedavg-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    found: list[Path] = []
    for rd in candidates:
        meta_path = rd / "run_metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = _json.loads(meta_path.read_text())
            if meta.get("seed") == target_seed:
                found.append(rd)
        except Exception:
            continue
    return found[0] if found else None


def _build_fake_artifacts(owner_id: str, code_length: int, seed: int, codebook_type: str = "multi-bit") -> dict:
    """Build a synthetic artifacts dict for verification against a clean model."""
    if codebook_type == "single-trigger":
        codebook = generate_single_trigger_codebook(code_length)
    elif codebook_type == "hadamard":
        owner_index = sum(ord(ch) for ch in owner_id) % code_length
        codebook = generate_hadamard_codebook(owner_index, code_length)
    else:
        codebook = generate_codebook(owner_id, code_length, seed)
    pos_queries = [q.tolist() for q in build_positive_queries(codebook, seed)]
    neg_queries = [q.tolist() for q in build_negative_queries(codebook, seed)]
    codebook_hash = build_codebook_hash(codebook)
    return {
        "owner_id": owner_id,
        "codebook": codebook,
        "codebook_hash": codebook_hash,
        "positive_queries": pos_queries,
        "negative_queries": neg_queries,
        "wm_train_config": {"seed": seed, "code_length": code_length, "codebook_type": codebook_type},
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/train",
    config_name="watermark_baseline",
)
def main(cfg: DictConfig) -> None:
    dataset = str(cfg.dataset.name)
    num_nonowners = int(cfg.get("num_nonowners", 20))
    output_root = Path(cfg.get("output_root", f"outputs/runs/{dataset}-fpr-nonowners"))
    tau = float(cfg.get("decision_threshold", 0.5))
    gamma = float(cfg.get("decision_margin", 0.05))
    negative_weight = float(cfg.get("negative_weight", 0.2))
    codebook_seed = int(cfg.get("fpr_codebook_seed", 0))
    codebook_type = str(cfg.get("fpr_codebook_type", "multi-bit"))
    competitor_ids = list(cfg.get("competitor_owner_ids", ["owner1", "owner2", "owner3", "owner4"]))
    code_length = int(cfg.watermarking.code_length)
    skip_training = bool(cfg.get("skip_training", False))

    if not skip_training:
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
                progress_enabled=bool(cfg.progress.enabled),
                device=str(cfg.device),
            )

    print(f"\nRunning verification on all {num_nonowners} non-owner models...")
    print(f"FPR evaluation: tau={tau}, gamma={gamma}")

    # Build artifacts once for all clean models (same codebook, seed=0)
    artifacts = _build_fake_artifacts("owner0", code_length, codebook_seed, codebook_type)
    owner_codebook_hash = artifacts["codebook_hash"]
    artifacts_path = output_root / "_fpr_artifacts.json"
    write_json(artifacts_path, artifacts)

    results = []
    for nonowner_seed in range(100, 100 + num_nonowners):
        run_dir = _find_run_dir_by_seed(output_root, nonowner_seed)
        if run_dir is None:
            print(f"  [skip] seed {nonowner_seed}: no run dir found")
            continue

        ckpt = run_dir / "best_checkpoint.pt"
        if not ckpt.exists():
            ckpt = run_dir / "checkpoint.pt"
        if not ckpt.exists():
            print(f"  [skip] seed {nonowner_seed}: no checkpoint")
            continue

        # Read acc for reporting
        metrics_path = run_dir / "metrics.json"
        acc_str = ""
        if metrics_path.exists():
            m = _json.loads(metrics_path.read_text())
            rlist = m.get("rounds", [])
            if rlist:
                best = max(r.get("val_accuracy", 0) for r in rlist)
                acc_str = f", acc={best:.4f}"

        summary = run_verification_from_checkpoint(
            checkpoint_path=ckpt,
            artifacts_path=artifacts_path,
            decision_threshold=tau,
            margin=gamma,
            competitor_owner_ids=competitor_ids,
            seed=0,
            hard_label_only=True,
            negative_weight=negative_weight,
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
              f"margin={results[-1]['margin_value']:.4f}, passed={passed}{acc_str}")

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
    print("\nThreshold sweep (tau, gamma -> FPR):")
    for row in threshold_sweep:
        if row["tau"] == tau:
            print(f"  tau={row['tau']:.2f}, gamma={row['gamma']:.2f}: "
                  f"FPR={row['fpr']:.4f} ({row['fp_count']}/{row['total']})")

    report_path = output_root / f"fpr_evaluation_report_{codebook_type}.json"
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
