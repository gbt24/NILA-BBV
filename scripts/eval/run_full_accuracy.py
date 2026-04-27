"""Evaluate full test-set accuracy for all checkpoints under a run directory.

Produces accuracy.json per run with:
  {seed, dataset, accuracy, num_samples, model_name, pre_score, neg_asr}
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bbv.datasets.loaders import load_dataset
from bbv.models import build_model


def _resolve_verification_file(run_dir: Path):
    for fname in [
        "verification_with_competitors_logits_seedmatched.json",
        "verification_margin_summary.json",
        "verification_summary.json",
    ]:
        path = run_dir / fname
        if path.exists():
            return path
    return None


def evaluate_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_single_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model_state")
    model_name = checkpoint.get("model_name", "resnet18")
    num_classes = checkpoint.get("num_classes", 10)

    # detected dataset from run_metadata
    metadata_path = run_dir / "run_metadata.json"
    dataset_name = "cifar10"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset_name = meta.get("dataset_name", meta.get("dataset", "cifar10"))

    # load dataset
    loaded = load_dataset(root=Path("data/raw"), train=False, download=True, name=dataset_name)
    dataset = loaded.dataset

    input_shape = checkpoint.get("input_shape")
    if input_shape is None:
        sample, _ = dataset[0]
        input_shape = tuple(int(d) for d in sample.shape)

    model = build_model(model_name=model_name, num_classes=num_classes, input_shape=input_shape)
    model.load_state_dict(model_state)
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    accuracy = evaluate_accuracy(model, loader, device)

    meta_path = run_dir / "run_metadata.json"
    seed = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seed = meta.get("seed", 0)

    vf_path = _resolve_verification_file(run_dir)
    pre_score = None
    neg_asr = None
    if vf_path:
        vdata = json.loads(vf_path.read_text(encoding="utf-8"))
        pre_score = vdata.get("owner_score")
        neg_asr = vdata.get("negative_asr")

    return {
        "run_dir": str(run_dir),
        "seed": seed,
        "dataset": dataset_name,
        "model_name": model_name,
        "num_classes": num_classes,
        "accuracy": accuracy,
        "num_samples": len(dataset),
        "owner_score": pre_score,
        "negative_asr": neg_asr,
    }


def main():
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/runs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # collect run directories recusively
    run_dirs = []
    for candidate in sorted(root.rglob("fedavg-*")):
        if candidate.is_dir() and ((candidate / "best_checkpoint.pt").exists() or (candidate / "checkpoint.pt").exists()):
            run_dirs.append(candidate)

    if not run_dirs:
        print(f"No run directories with checkpoints found under {root}")
        return

    results = []
    for run_dir in run_dirs:
        checkpoint_path = run_dir / "best_checkpoint.pt"
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            print(f"  SKIP {run_dir.name}: no checkpoint")
            continue

        print(f"  Evaluating {run_dir.name}...")
        try:
            result = evaluate_single_checkpoint(checkpoint_path, run_dir, device)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue
        results.append(result)
        print(f"    {result['dataset']} seed={result['seed']} acc={result['accuracy']:.4f}")

        # also write a per-run file so downstream scripts can read it
        out_path = run_dir / "accuracy.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write aggregated results
    agg_path = root / "full_accuracy.json"
    agg_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAggregated results written to {agg_path}")
    print(f"Evaluated {len(results)} checkpoints.")


if __name__ == "__main__":
    main()
