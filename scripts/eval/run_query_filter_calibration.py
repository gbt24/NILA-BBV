"""Build a clean-calibrated query subset for one owner run.

This script scores each positive query bit against a pool of clean models,
then keeps the top-k least-triggered bits. It can optionally report owner-side
bit recovery on the owner model for sanity checking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bbv.verification.baseline import (
    _load_model_from_checkpoint,
    _resolve_verification_device,
    batched_query_model_logits,
    load_owner_artifacts,
    recover_codeword_from_logits,
)
from bbv.watermarking.codebook import generate_codebook
from bbv.utils.io import write_json


def _latest_runs_by_seed(output_root: Path) -> list[Path]:
    by_seed: dict[int, Path] = {}
    for run_dir in output_root.glob("fedavg-*"):
        meta_path = run_dir / "run_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        seed = int(meta["seed"])
        current = by_seed.get(seed)
        if current is None or run_dir.stat().st_mtime > current.stat().st_mtime:
            by_seed[seed] = run_dir
    return [by_seed[seed] for seed in sorted(by_seed)]


def _checkpoint_path(run_dir: Path) -> Path:
    best = run_dir / "best_checkpoint.pt"
    return best if best.exists() else run_dir / "checkpoint.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-run-dir", required=True)
    parser.add_argument("--clean-output-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=["clean-only", "owner-priority", "owner-minus-competitor"],
        default="clean-only",
        help="Bit ranking strategy.",
    )
    parser.add_argument(
        "--competitor-owner-ids",
        nargs="+",
        default=["owner1", "owner2", "owner3", "owner4"],
        help="Competitor owner IDs for owner-minus-competitor mode.",
    )
    parser.add_argument(
        "--max-clean-rate",
        type=float,
        default=1.0,
        help="Discard bits with clean_match_rate above this threshold before ranking.",
    )
    args = parser.parse_args()

    owner_run_dir = Path(args.owner_run_dir)
    clean_output_root = Path(args.clean_output_root)
    output_path = Path(args.output_path)

    artifacts = load_owner_artifacts(owner_run_dir / "owner_artifacts.json")
    expected = [int(bit) for bit in artifacts["codebook"]]
    pos_queries = [artifacts["positive_queries"][i] for i in range(len(expected))]
    code_length = len(expected)
    if args.top_k <= 0 or args.top_k > code_length:
        raise ValueError(f"top_k must be in [1, {code_length}]")
    if not 0.0 <= args.max_clean_rate <= 1.0:
        raise ValueError("max_clean_rate must be in [0, 1]")
    if args.mode == "owner-minus-competitor" and str(artifacts.get("owner_id", "")) in args.competitor_owner_ids:
        raise ValueError("owner must not appear in --competitor-owner-ids")

    verification_device = _resolve_verification_device(args.device)

    owner_model = _load_model_from_checkpoint(_checkpoint_path(owner_run_dir), verification_device)
    owner_logits = batched_query_model_logits(
        model=owner_model,
        queries=[torch.tensor(sample, dtype=torch.float32) for sample in pos_queries],
        batch_size=args.batch_size,
    )
    owner_recovered = recover_codeword_from_logits(owner_logits)
    owner_match = [int(a == b) for a, b in zip(owner_recovered, expected)]

    competitor_match_rate: list[float] = [0.0] * code_length
    if args.mode == "owner-minus-competitor":
        run_meta = json.loads((owner_run_dir / "run_metadata.json").read_text())
        cb_seed = int(run_meta["seed"])
        competitor_codebooks = [
            generate_codebook(owner_id=cid, code_length=code_length, seed=cb_seed)
            for cid in args.competitor_owner_ids
        ]
        for i in range(code_length):
            comp_hits = sum(int(owner_recovered[i] == cb[i]) for cb in competitor_codebooks)
            competitor_match_rate[i] = comp_hits / len(competitor_codebooks)

    clean_runs = _latest_runs_by_seed(clean_output_root)
    if not clean_runs:
        raise FileNotFoundError(f"no clean runs found under {clean_output_root}")

    clean_match_counts = [0] * code_length
    for run_dir in clean_runs:
        model = _load_model_from_checkpoint(_checkpoint_path(run_dir), verification_device)
        logits = batched_query_model_logits(
            model=model,
            queries=[torch.tensor(sample, dtype=torch.float32) for sample in pos_queries],
            batch_size=args.batch_size,
        )
        recovered = recover_codeword_from_logits(logits)
        for i, (pred, exp) in enumerate(zip(recovered, expected)):
            clean_match_counts[i] += int(pred == exp)

    bit_stats = []
    clean_n = len(clean_runs)
    for i in range(code_length):
        clean_rate = clean_match_counts[i] / clean_n
        bit_stats.append(
            {
                "bit": i,
                "clean_match_rate": clean_rate,
                "owner_match": owner_match[i],
                "competitor_match_rate": competitor_match_rate[i],
            }
        )

    eligible = [row for row in bit_stats if row["clean_match_rate"] <= args.max_clean_rate]
    pool = eligible if len(eligible) >= args.top_k else bit_stats
    if args.mode == "owner-minus-competitor":
        ranked = sorted(pool, key=lambda row: (
            -(row["owner_match"] - row["competitor_match_rate"]),
            row["clean_match_rate"],
            row["bit"],
        ))
    elif args.mode == "owner-priority":
        ranked = sorted(pool, key=lambda row: (-row["owner_match"], row["clean_match_rate"], row["bit"]))
    else:
        ranked = sorted(pool, key=lambda row: (row["clean_match_rate"], -row["owner_match"], row["bit"]))
    selected_indices = [int(row["bit"]) for row in ranked[: args.top_k]]
    selected_owner_matches = sum(int(bit_stats[index]["owner_match"]) for index in selected_indices)
    selected_comp_hits = sum(bit_stats[index]["competitor_match_rate"] for index in selected_indices) * len(args.competitor_owner_ids)

    payload = {
        "owner_run_dir": str(owner_run_dir),
        "clean_output_root": str(clean_output_root),
        "top_k": args.top_k,
        "mode": args.mode,
        "competitor_owner_ids": list(args.competitor_owner_ids),
        "max_clean_rate": args.max_clean_rate,
        "owner_id": str(artifacts["owner_id"]),
        "code_length": code_length,
        "codebook_hash": str(artifacts.get("codebook_hash", "")),
        "selected_indices": selected_indices,
        "selected_owner_matches": selected_owner_matches,
        "num_clean_runs": clean_n,
        "bit_stats": bit_stats,
    }
    write_json(output_path, payload)
    print(f"Wrote calibration to {output_path}")
    print(f"Selected {len(selected_indices)} / {code_length} bits")
    print(f"Mode: {args.mode} | owner-matched bits retained: {selected_owner_matches}/{len(selected_indices)}")
    if args.mode == "owner-minus-competitor":
        avg_comp_rate = selected_comp_hits / (len(selected_indices) * len(args.competitor_owner_ids)) if len(selected_indices) > 0 else 0
        print(f"Competitor match rate on selected bits: {avg_comp_rate:.3f}")
    print(f"Eligible bits under clean-rate ceiling: {len(eligible)}/{code_length}")


if __name__ == "__main__":
    main()
