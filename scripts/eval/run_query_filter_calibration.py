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

    verification_device = _resolve_verification_device(args.device)

    owner_model = _load_model_from_checkpoint(_checkpoint_path(owner_run_dir), verification_device)
    owner_logits = batched_query_model_logits(
        model=owner_model,
        queries=[torch.tensor(sample, dtype=torch.float32) for sample in pos_queries],
        batch_size=args.batch_size,
    )
    owner_recovered = recover_codeword_from_logits(owner_logits)
    owner_match = [int(a == b) for a, b in zip(owner_recovered, expected)]

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
            }
        )

    ranked = sorted(bit_stats, key=lambda row: (row["clean_match_rate"], -row["owner_match"], row["bit"]))
    selected_indices = [int(row["bit"]) for row in ranked[: args.top_k]]

    payload = {
        "owner_run_dir": str(owner_run_dir),
        "clean_output_root": str(clean_output_root),
        "top_k": args.top_k,
        "owner_id": str(artifacts["owner_id"]),
        "code_length": code_length,
        "codebook_hash": str(artifacts.get("codebook_hash", "")),
        "selected_indices": selected_indices,
        "num_clean_runs": clean_n,
        "bit_stats": bit_stats,
    }
    write_json(output_path, payload)
    print(f"Wrote calibration to {output_path}")
    print(f"Selected {len(selected_indices)} / {code_length} bits")


if __name__ == "__main__":
    main()
