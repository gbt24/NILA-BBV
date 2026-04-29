#!/usr/bin/env python3
"""Generate publication-ready figures and tables for the NILA-BBV paper.

Figures:
  Fig 1 – Method framework diagram (static)
  Fig 2 – Main results: owner score + competitor margin (CIFAR-10 + MNIST)
  Fig 3 – Attack robustness: pre/post owner score (dual panel)
  Fig 4 – Non-IID effects: Dirichlet + quantity skew (CIFAR-10)
  Fig 5 – Owner vs competitor score distribution
  Fig 6 – Training convergence curves (loss + accuracy per round)

Tables (CSV):
  table-main-results.csv
  table-ablation.csv
  table-attack-robustness.csv
"""

import json
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Global style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(PROJECT_ROOT, "outputs", "runs")
ATTACKS_DIR = os.path.join(PROJECT_ROOT, "outputs", "attacks")
FIG_DIR = os.path.join(PROJECT_ROOT, "analysis-output", "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "analysis-output", "tables")

THRESHOLD = 0.5
MARGIN = 0.05

COLORS = {
    "cifar10":       "#2E86AB",
    "mnist":         "#E76F51",
    "cifar100":      "#A23B72",
    "threshold":     "#B91C1C",
    "competitor":    "#9CA3AF",
    "owner":         "#2E86AB",
    "pass":          "#059669",
    "fail":          "#DC2626",
    "finetune":      "#7CB518",
    "pruning":       "#A23B72",
    "quantization":  "#2E86AB",
    "distillation":  "#F18F01",
    "extraction":    "#C73E1D",
    "adaptive":      "#2E86AB",
    "uniform":       "#9CA3AF",
}

DEFAULT_VERIFICATION_FILES = [
    "verification_with_competitors_logits_seedmatched.json",
    "verification_margin_summary.json",
    "verification_summary.json",
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _find_verification_file(run_dir_path):
    for fname in DEFAULT_VERIFICATION_FILES:
        path = os.path.join(run_dir_path, fname)
        if os.path.exists(path):
            return path
    return None

def _read_run_verifications(run_dir, seed_dirs=None):
    """Return list of verification dicts for each seed subdir."""
    results = []
    if seed_dirs is None:
        seed_dirs = sorted(d for d in os.listdir(run_dir)
                           if os.path.isdir(os.path.join(run_dir, d)))
    for sd in seed_dirs:
        path = _find_verification_file(os.path.join(run_dir, sd))
        if path:
            with open(path) as f:
                results.append(json.load(f))
    return results

def _extract_competitor_max(vdata):
    if "competitor_scores" in vdata and vdata["competitor_scores"]:
        return max(vdata["competitor_scores"].values())
    return 0.0

def _mean_std(values):
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))

def save_figure(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in [".png", ".pdf"]:
        path = os.path.join(FIG_DIR, f"{name}{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  [fig] {name}.png + .pdf")

def save_table(data, name, headers):
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = os.path.join(TABLE_DIR, name)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)
    print(f"  [table] {name}")

def _draw_module(ax, x, y, w, h, title, lines, inp, out, phase):
    color = "#2E86AB" if phase == "train" else "#B91C1C"
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        facecolor="white", edgecolor=color, linewidth=1.8,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.14, title,
            ha="center", va="top", fontsize=9, fontweight="bold", color=color)
    ax.plot([x + 0.12, x + w - 0.12], [y + h - 0.30, y + h - 0.30],
            color=color, linewidth=0.7, alpha=0.5)
    line_y = y + h - 0.42
    for line in lines:
        ax.text(x + 0.14, line_y, line, ha="left", va="top",
                fontsize=7.5, color="#333333")
        line_y -= 0.26
    ax.text(x + 0.1, y - 0.12, inp, ha="left", va="top",
            fontsize=6.5, color="#666666", style="italic")
    ax.text(x + w - 0.1, y - 0.12, out, ha="right", va="top",
            fontsize=6.5, color="#666666", style="italic")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 – Method framework
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig1_framework():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    train_bg = FancyBboxPatch(
        (0.25, 3.15), 11.5, 2.45, boxstyle="round,pad=0.1",
        facecolor="#E8F4F8", edgecolor="#2E86AB", linewidth=2, alpha=0.25)
    ax.add_patch(train_bg)
    ax.text(0.45, 5.40, "Training Phase", fontsize=11, fontweight="bold",
            color="#2E86AB", style="italic")

    verify_bg = FancyBboxPatch(
        (0.25, 0.25), 11.5, 2.45, boxstyle="round,pad=0.1",
        facecolor="#FDE8E8", edgecolor="#B91C1C", linewidth=2, alpha=0.25)
    ax.add_patch(verify_bg)
    ax.text(0.45, 2.50, "Verification Phase", fontsize=11, fontweight="bold",
            color="#B91C1C", style="italic")

    modules = [
        (0.55, 3.45, 2.45, 1.55, "Codebook & Queries", [
            r"Owner codeword $\mathbf{c}_i$",
            r"Query set $Q_i$",
            r"Diagnostic negative set $Q_i^-$",
        ], "In: auxiliary data", r"Out: $\{\mathbf{c}_i,Q_i,Q_i^-\}$", "train"),
        (3.45, 3.45, 2.45, 1.55, "Non-IID Scoring", [
            r"Gradient conflict",
            r"Label coverage",
            r"Client score $a_k$",
        ], "In: client statistics", "Out: allocation weights", "train"),
        (6.35, 3.45, 2.45, 1.55, "Adaptive Embedding", [
            r"FL training with $\mathcal{L}_{wm}$",
            r"Client-specific weights",
            r"Aggregate model $\theta$",
        ], "In: codebook, $a_k$", "Out: watermarked $\theta$", "train"),
        (9.25, 3.45, 2.05, 1.55, "Commitment", [
            r"Owner ID",
            r"Codebook hash",
            r"Training config hash",
        ], "In: evidence metadata", "Out: audit record", "train"),
        (1.55, 0.65, 3.05, 1.55, "Black-box Querying", [
            r"Query suspicious model",
            r"Recover codeword $\hat{\mathbf{c}}_i$",
            r"Compute owner score $s_i$",
        ], "In: model API, $Q_i$", "Out: scores", "verify"),
        (6.65, 0.65, 3.65, 1.55, "Margin Decision", [
            r"Threshold: $s_i \geq \tau$",
            r"Margin: $s_i - \max_{j\ne i}s_j \geq \gamma$",
            r"Verify / Reject / Ambiguous",
        ], "In: scores, commitment", "Out: ownership decision", "verify"),
    ]

    for m in modules:
        _draw_module(ax, *m)

    for i in range(3):
        x1 = modules[i][0] + modules[i][2]
        y_mid = modules[i][1] + modules[i][3] / 2
        x2 = modules[i + 1][0]
        ax.annotate("", xy=(x2 - 0.06, y_mid), xytext=(x1 + 0.06, y_mid),
                     arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    x_theta = modules[2][0] + modules[2][2] / 2
    y_top = modules[2][1]
    y_bot = modules[4][1] + modules[4][3]
    ax.annotate("", xy=(modules[4][0] + modules[4][2] / 2, y_bot + 0.06),
                 xytext=(x_theta, y_top - 0.06),
                 arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=2.2,
                                 connectionstyle="arc3,rad=-0.28"))
    ax.text(4.8, 2.85, r"Model API $\theta$",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color="#B91C1C")

    x1 = modules[4][0] + modules[4][2]
    y_mid = modules[4][1] + modules[4][3] / 2
    x2 = modules[5][0]
    ax.annotate("", xy=(x2 - 0.06, y_mid), xytext=(x1 + 0.06, y_mid),
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    x_commit = modules[3][0] + modules[3][2] / 2
    ax.annotate("", xy=(modules[5][0] + modules[5][2] / 2, modules[5][1] + modules[5][3] + 0.06),
                 xytext=(x_commit, modules[3][1] - 0.06),
                 arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=1.8,
                                 connectionstyle="arc3,rad=0.25"))
    ax.text(8.7, 2.75, "Commitment check", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#7C3AED")

    ax.set_title("NILA-BBV Framework Overview",
                 fontsize=12, fontweight="bold", pad=15)
    save_figure(fig, "figure-01-framework")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 – Main results (owner + competitor margin)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig2_main_results():
    datasets = [
        ("CIFAR-10\nMulti-bit", os.path.join(RUNS_DIR, "cifar10-main-adaptive-tuneB"), "cifar10"),
        ("CIFAR-10\nHadamard",  os.path.join(RUNS_DIR, "cifar10-hadamard"),              "cifar10"),
        ("CIFAR-10\nSingle-trig", os.path.join(RUNS_DIR, "cifar10-single-trigger-baseline"), "cifar10"),
        ("MNIST",               os.path.join(RUNS_DIR, "mnist-main-adaptive"),           "mnist"),
    ]

    names = []
    owner_means, owner_stds = [], []
    comp_means, comp_stds = [], []
    pass_rates = []

    for name, path, key in datasets:
        data = _read_run_verifications(path)
        if not data:
            continue
        scores = [d["owner_score"] for d in data]
        comps = [max(d.get("competitor_scores", {}).values()) if d.get("competitor_scores") else 0.0
                 for d in data]
        decisions = [d["decision"] for d in data]
        names.append(name)
        owner_means.append(np.mean(scores))
        owner_stds.append(np.std(scores, ddof=1))
        comp_means.append(np.mean(comps))
        comp_stds.append(np.std(comps, ddof=1))
        pass_rates.append(f"{sum(int(bool(d)) for d in decisions)}/{len(decisions)}")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(names))
    width = 0.30

    bars_o = ax.bar(x - width / 2, owner_means, width, yerr=owner_stds,
                     capsize=6, color=COLORS["owner"], alpha=0.9,
                     edgecolor="white", linewidth=0.8,
                     error_kw={"lw": 1.2}, label="Owner")
    bars_c = ax.bar(x + width / 2, comp_means, width, yerr=comp_stds,
                     capsize=6, color=COLORS["competitor"], alpha=0.7,
                     edgecolor="white", linewidth=0.8,
                     error_kw={"lw": 1.2}, label="Max competitor")

    ax.axhline(y=THRESHOLD, color=COLORS["threshold"], linestyle="--",
               linewidth=1.5, label=f"Decision threshold ($\\tau={THRESHOLD}$)")

    for i, (bar, pr) in enumerate(zip(bars_o, pass_rates)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                pr, ha="center", fontsize=10, fontweight="bold",
                color=COLORS["pass"] if "3/3" in pr else COLORS["fail"])

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Main Results: Owner vs Competitor Verification\n(CIFAR-10 with Three Codebook Designs + MNIST)",
                 fontweight="bold", fontsize=10)
    save_figure(fig, "figure-02-main-results")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 – Attack robustness (triple panel)
# ══════════════════════════════════════════════════════════════════════════════

def _collect_attack_scores(attacks_root, attack_types, seeds):
    """Return {atk: [scores]} across all seeds."""
    import glob as _glob
    scores = {atk: [] for atk in attack_types}
    decisions = {atk: [] for atk in attack_types}
    for seed in seeds:
        seed_dir = f"{attacks_root}-seed{seed}"
        for atk in attack_types:
            pattern = os.path.join(ATTACKS_DIR, seed_dir, f"{atk}-*",
                                    "verification_after_attack.json")
            matches = _glob.glob(pattern)
            if matches:
                with open(matches[0]) as f:
                    d = json.load(f)
                scores[atk].append(d["owner_score"])
                decisions[atk].append(d["decision"])
    return scores, decisions


def _collect_attack_scores_flat(attacks_root, attack_types, seeds):
    """Same as _collect_attack_scores but for flat dir structure (no -seedN suffix)."""
    import glob as _glob
    scores = {atk: [] for atk in attack_types}
    decisions = {atk: [] for atk in attack_types}
    root_dir = os.path.join(ATTACKS_DIR, attacks_root)
    for atk in attack_types:
        pattern = os.path.join(root_dir, f"{atk}-*", "verification_after_attack.json")
        matches = sorted(_glob.glob(pattern))
        for m in matches[:len(seeds)]:  # take first N = num seeds (time-ordered)
            with open(m) as f:
                d = json.load(f)
            scores[atk].append(d["owner_score"])
            decisions[atk].append(d["decision"])
    return scores, decisions

def plot_fig3_attack_robustness():
    attack_types = ["finetune", "quantization"]
    attack_labels = ["Finetune", "Quantization"]
    attack_colors = [COLORS[a] for a in attack_types]
    seeds = [0, 1, 2]

    # CIFAR-10 Random multi-bit
    c10r_scores, c10r_dec = _collect_attack_scores("cifar10-robustness", attack_types, seeds)
    # CIFAR-10 Hadamard
    c10h_scores, c10h_dec = _collect_attack_scores_flat("cifar10-hadamard-robustness", attack_types, seeds)
    # MNIST
    mn_scores, mn_dec = _collect_attack_scores("mnist-robustness", attack_types, seeds)

    # Pre-attack owner scores (mean)
    c10r_pre = _read_run_verifications(os.path.join(RUNS_DIR, "cifar10-main-adaptive-tuneB"))
    c10h_pre = _read_run_verifications(os.path.join(RUNS_DIR, "cifar10-hadamard"))
    mn_pre = _read_run_verifications(os.path.join(RUNS_DIR, "mnist-main-adaptive"))
    c10r_pre_mean = np.mean([d["owner_score"] for d in c10r_pre])
    c10h_pre_mean = np.mean([d["owner_score"] for d in c10h_pre])
    mn_pre_mean = np.mean([d["owner_score"] for d in mn_pre])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (scores, decisions), pre_mean, ds_name, cb_label in [
        (ax1, (c10r_scores, c10r_dec), c10r_pre_mean, "CIFAR-10", "Random"),
        (ax2, (c10h_scores, c10h_dec), c10h_pre_mean, "CIFAR-10", "Hadamard"),
        (ax3, (mn_scores, mn_dec), mn_pre_mean, "MNIST", "Random"),
    ]:
        x = np.arange(len(attack_labels))
        means = [np.mean(scores[a]) for a in attack_types]
        stds = [np.std(scores[a], ddof=1) for a in attack_types]
        pass_rates = [f"{sum(bool(int(d)) for d in decisions[a])}/{len(decisions[a])}"
                      for a in attack_types]

        bars = ax.bar(x, means, yerr=stds, capsize=6, width=0.5,
                       color=attack_colors, edgecolor="white", linewidth=0.8,
                       error_kw={"lw": 1.2})
        ax.axhline(y=pre_mean, color=COLORS["owner"], linestyle="--", linewidth=1.5,
                    label=f"Pre-attack ({pre_mean:.3f})")
        ax.axhline(y=THRESHOLD, color=COLORS["threshold"], linestyle=":",
                    linewidth=1.2, alpha=0.6)

        for bar, pr in zip(bars, pass_rates):
            color = COLORS["pass"] if "3/" in pr else COLORS["fail"]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    pr, ha="center", fontsize=8.5, fontweight="bold", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(attack_labels, fontsize=8.5, rotation=25, ha="right")
        ax.set_ylabel("Post-attack owner score" if ax == ax1 else "", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=7.5)
        panel_label = chr(97 + [ax1, ax2, ax3].index(ax))
        ax.set_title(f"({panel_label}) {ds_name}\n{cb_label} codebook",
                     fontweight="bold", fontsize=10)

    fig.suptitle("Attack Robustness: Post-Attack Owner Verification Scores",
                 fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure-03-attack-robustness")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 – Non-IID effects (CIFAR-10: Hadamard Dirichlet label skew)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig4_non_iid():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    # Helper to read accuracy
    def _read_acc(run_dir):
        accs = []
        for sd in sorted(os.listdir(run_dir)):
            sp = os.path.join(run_dir, sd)
            mp = os.path.join(sp, "metrics.json")
            if os.path.exists(mp):
                with open(mp) as f:
                    d = json.load(f)
                if "rounds" in d and len(d["rounds"]) > 0:
                    best = max(r.get("val_accuracy", 0) for r in d["rounds"])
                    if best > 0:
                        accs.append(best * 100)
        return np.mean(accs) if accs else None

    # IID reference
    iid_dir = os.path.join(RUNS_DIR, "cifar10-hadamard")
    iid_data = _read_run_verifications(iid_dir)
    iid_scores = [d["owner_score"] for d in iid_data]
    iid_mean = np.mean(iid_scores)

    # ── Panel (a): Dirichlet label skew ──
    alphas = ["0.1", "0.3", "0.5", "1.0"]
    dir_means, dir_stds, dir_pass, dir_accs = [], [], [], []
    for a in alphas:
        rdir = os.path.join(RUNS_DIR, f"cifar10-hadamard-dirichlet-a{a}")
        if not os.path.isdir(rdir): continue
        data = _read_run_verifications(rdir)
        scores = [d["owner_score"] for d in data]
        decisions = [d["decision"] for d in data]
        dir_means.append(np.mean(scores))
        dir_stds.append(np.std(scores, ddof=1) if len(scores) > 1 else 0.0)
        dir_pass.append(f"{sum(int(bool(d)) for d in decisions)}/{len(decisions)}")
        acc = _read_acc(rdir)
        if acc is not None: dir_accs.append(acc)

    x = np.arange(len(alphas))
    bars = ax1.bar(x, dir_means, yerr=dir_stds, capsize=6, width=0.5,
                    color="#7CB518", edgecolor="white", linewidth=0.8,
                    error_kw={"lw": 1.2})
    ax1.axhline(y=iid_mean, color=COLORS["owner"], linestyle="--", linewidth=1.3,
                 alpha=0.7)
    ax1.axhline(y=THRESHOLD, color=COLORS["threshold"], linestyle=":", linewidth=1.2,
                 alpha=0.6)
    for bar, pr in zip(bars, dir_pass):
        color = COLORS["pass"] if "3/" in pr else COLORS["fail"]
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                 pr, ha="center", fontsize=9, fontweight="bold", color=color)
    if len(dir_accs) == len(dir_means):
        for bar, acc in zip(bars, dir_accs):
            ax1.text(bar.get_x() + bar.get_width() / 2, 0.03,
                     f"{acc:.0f}%", ha="center", fontsize=7, color="#666666")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\alpha$={a}" for a in alphas], fontsize=8.5)
    ax1.set_ylabel("Mean owner score", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(a) Dirichlet Label Skew", fontweight="bold", fontsize=9.5)
    ax1.text(0.98, 0.95, f"IID={iid_mean:.3f}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=7, color=COLORS["owner"], style="italic")

    # ── Panel (b): Quantity skew ──
    sigmas = ["0.5", "1.0"]
    q_means, q_stds, q_pass, q_accs = [], [], [], []
    for s in sigmas:
        rdir = os.path.join(RUNS_DIR, f"cifar10-hadamard-quantity-s{s}")
        if not os.path.isdir(rdir): continue
        data = _read_run_verifications(rdir)
        scores = [d["owner_score"] for d in data]
        decisions = [d["decision"] for d in data]
        q_means.append(np.mean(scores))
        q_stds.append(np.std(scores, ddof=1) if len(scores) > 1 else 0.0)
        q_pass.append(f"{sum(int(bool(d)) for d in decisions)}/{len(decisions)}")
        acc = _read_acc(rdir)
        if acc is not None: q_accs.append(acc)

    x2 = np.arange(len(sigmas))
    bars2 = ax2.bar(x2, q_means, yerr=q_stds, capsize=6, width=0.4,
                     color="#C73E1D", edgecolor="white", linewidth=0.8,
                     error_kw={"lw": 1.2})
    ax2.axhline(y=iid_mean, color=COLORS["owner"], linestyle="--", linewidth=1.3,
                 alpha=0.7)
    ax2.axhline(y=THRESHOLD, color=COLORS["threshold"], linestyle=":", linewidth=1.2,
                 alpha=0.6)
    for bar, pr in zip(bars2, q_pass):
        color = COLORS["pass"] if "3/" in pr else COLORS["fail"]
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                 pr, ha="center", fontsize=9, fontweight="bold", color=color)
    if len(q_accs) == len(q_means):
        for bar, acc in zip(bars2, q_accs):
            ax2.text(bar.get_x() + bar.get_width() / 2, 0.03,
                     f"{acc:.0f}%", ha="center", fontsize=7, color="#666666")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"$\\sigma$={s}" for s in sigmas], fontsize=8.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) Quantity Skew", fontweight="bold", fontsize=9.5)
    ax2.text(0.98, 0.95, f"IID={iid_mean:.3f}", transform=ax2.transAxes,
             ha="right", va="top", fontsize=7, color=COLORS["owner"], style="italic")

    fig.suptitle("Non-IID Sensitivity: Hadamard Codebook (CIFAR-10)",
                 fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure-04-non-iid")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 – Owner vs competitor score distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig5_competitor_distribution():
    datasets = [
        ("CIFAR-10\nMulti-bit",  os.path.join(RUNS_DIR, "cifar10-main-adaptive-tuneB"), "cifar10"),
        ("CIFAR-10\nHadamard",   os.path.join(RUNS_DIR, "cifar10-hadamard"),              "cifar10"),
        ("CIFAR-10\nSingle-trig", os.path.join(RUNS_DIR, "cifar10-single-trigger-baseline"), "cifar10"),
        ("MNIST",                os.path.join(RUNS_DIR, "mnist-main-adaptive"),           "mnist"),
    ]

    names = []
    owner_data, competitor_data = [], []

    for name, path, key in datasets:
        data = _read_run_verifications(path)
        if not data:
            continue
        names.append(name)
        owners = [d["owner_score"] for d in data]
        competitors = []
        for d in data:
            for v in d.get("competitor_scores", {}).values():
                competitors.append(v)
        owner_data.append(owners)
        competitor_data.append(competitors)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    n = len(names)
    width = 0.30

    for i, name in enumerate(names):
        bp_o = ax.boxplot(
            owner_data[i], positions=[i - width / 2], widths=width * 0.7,
            patch_artist=True, medianprops={"color": "black", "linewidth": 1.2})
        bp_o["boxes"][0].set_facecolor(COLORS["owner"])
        bp_o["boxes"][0].set_alpha(0.85)

        bp_c = ax.boxplot(
            competitor_data[i], positions=[i + width / 2], widths=width * 0.7,
            patch_artist=True, medianprops={"color": "black", "linewidth": 1.2})
        bp_c["boxes"][0].set_facecolor(COLORS["competitor"])
        bp_c["boxes"][0].set_alpha(0.55)

        np.random.seed(42 + i)
        jitter_o = np.random.normal(i - width / 2, 0.02, len(owner_data[i]))
        ax.scatter(jitter_o, owner_data[i], s=20, c=COLORS["owner"],
                   edgecolors="white", linewidth=0.5, zorder=5)
        jitter_c = np.random.normal(i + width / 2, 0.04, len(competitor_data[i]))
        ax.scatter(jitter_c, competitor_data[i], s=6, c="#666666",
                   edgecolors="white", linewidth=0.3, zorder=5)

    ax.axhline(y=THRESHOLD, color=COLORS["threshold"], linestyle="--", linewidth=1.5)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.05)

    legend_patches = [
        plt.matplotlib.patches.Patch(color=COLORS["owner"], alpha=0.85, label="Owner"),
        plt.matplotlib.patches.Patch(color=COLORS["competitor"], alpha=0.55,
                                      label="Competitors"),
        plt.Line2D([0], [0], color=COLORS["threshold"], linestyle="--",
                    label=f"Threshold ($\\tau={THRESHOLD}$)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7.5)

    ax.set_title("Owner vs Competitor Score Distributions (CIFAR-10 codebooks + MNIST)",
                 fontweight="bold", fontsize=10)
    save_figure(fig, "figure-05-competitor-distribution")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 – Training convergence curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig6_training_convergence():
    datasets = [
        ("CIFAR-10", os.path.join(RUNS_DIR, "cifar10-main-adaptive-tuneB"), "cifar10"),
        ("MNIST",    os.path.join(RUNS_DIR, "mnist-main-adaptive"),          "mnist"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    colors = [COLORS["cifar10"], COLORS["mnist"]]

    for ds_idx, (ds_name, path, key) in enumerate(datasets):
        seed_dirs = sorted(d for d in os.listdir(path)
                          if os.path.isdir(os.path.join(path, d)))
        for sd in seed_dirs:
            mpath = os.path.join(path, sd, "metrics.json")
            if not os.path.exists(mpath):
                continue
            with open(mpath) as f:
                metrics = json.load(f)
            rounds_data = metrics.get("rounds", [])
            if not rounds_data:
                continue
            rounds = [r["round"] for r in rounds_data]
            task_loss = [r.get("task_loss", 0) for r in rounds_data]
            wm_loss = [r.get("wm_loss", 0) for r in rounds_data]
            val_acc = [r.get("val_accuracy", 0) for r in rounds_data]

            ax = axes[ds_idx][0]
            ax.plot(rounds, task_loss, color=colors[ds_idx], alpha=0.7, linewidth=0.8)
            ax.set_title(f"({chr(97 + ds_idx * 3)}) {ds_name}: Task Loss", fontweight="bold", fontsize=9)
            ax.set_ylabel("Cross-entropy loss", fontsize=8)
            ax.set_xlabel("Round", fontsize=8)

            ax = axes[ds_idx][1]
            ax.plot(rounds, wm_loss, color=colors[ds_idx], alpha=0.7, linewidth=0.8)
            ax.set_title(f"({chr(97 + ds_idx * 3 + 1)}) {ds_name}: WM Loss", fontweight="bold", fontsize=9)
            ax.set_ylabel("WM loss", fontsize=8)
            ax.set_xlabel("Round", fontsize=8)

            ax = axes[ds_idx][2]
            ax.plot(rounds, val_acc, color=colors[ds_idx], alpha=0.7, linewidth=0.8)
            ax.set_title(f"({chr(97 + ds_idx * 3 + 2)}) {ds_name}: Val Accuracy", fontweight="bold", fontsize=9)
            ax.set_ylabel("Accuracy", fontsize=8)
            ax.set_xlabel("Round", fontsize=8)

    fig.suptitle("Training Convergence Curves (All Seeds)", fontweight="bold", fontsize=11, y=1.01)
    fig.tight_layout()
    save_figure(fig, "figure-06-training-convergence")


# ══════════════════════════════════════════════════════════════════════════════
# Tables
# ══════════════════════════════════════════════════════════════════════════════

def _table_main_results():
    rows = []
    for ds_label, ds_path in [
        ("CIFAR-10 (Multi-bit)", os.path.join(RUNS_DIR, "cifar10-main-adaptive-tuneB")),
        ("CIFAR-10 (Hadamard)",  os.path.join(RUNS_DIR, "cifar10-hadamard")),
        ("CIFAR-10 (Single-trig)", os.path.join(RUNS_DIR, "cifar10-single-trigger-baseline")),
        ("MNIST",                os.path.join(RUNS_DIR, "mnist-main-adaptive")),
    ]:
        data = _read_run_verifications(ds_path)
        scores = [d["owner_score"] for d in data]
        margins = [d.get("margin_value", 0) for d in data]
        neg_asrs = [d.get("negative_asr", 0) for d in data]
        decisions = [str(d["decision"]) for d in data]
        competitors = [(d.get("competitor_scores", {})) for d in data]
        comp_max = [max(c.values()) if c else 0 for c in competitors]
        m, s = _mean_std(scores)
        mm, sm = _mean_std(margins)
        ma, sa = _mean_std(neg_asrs)
        mc, sc = _mean_std(comp_max)
        rows.append([ds_label,
                      f"{m:.4f} ± {s:.4f}",
                      f"{mm:.4f} ± {sm:.4f}",
                      f"{mc:.4f} ± {sc:.4f}",
                      f"{ma:.3f} ± {sa:.3f}",
                      f"{sum(int(d == 'True') for d in decisions)}/{len(decisions)}",
                      ", ".join(f"{s:.3f}" for s in scores)])

    save_table(rows, "table-main-results.csv",
               ["Dataset", "Owner Score (mean±std)", "Margin (mean±std)",
                "Max Competitor (mean±std)", "Negative ASR (mean±std)",
                "Pass Rate", "Per-seed Scores"])


def _table_ablation():
    rows = []
    for ds_label, adaptive_path, uniform_path in [
        ("CIFAR-10",
         os.path.join(RUNS_DIR, "cifar10-main-adaptive-tuneB"),
         os.path.join(RUNS_DIR, "cifar10-ablation-off-tuneB")),
        ("MNIST",
         os.path.join(RUNS_DIR, "mnist-main-adaptive"),
         os.path.join(RUNS_DIR, "mnist-ablation-off")),
    ]:
        ada_data = _read_run_verifications(adaptive_path)
        uni_data = _read_run_verifications(uniform_path)
        ada_acc = _read_accuracy_json(adaptive_path)
        uni_acc = _read_accuracy_json(uniform_path)
        ada_scores = [d["owner_score"] for d in ada_data]
        uni_scores = [d["owner_score"] for d in uni_data]
        ada_accs = [r["accuracy"] for r in ada_acc] if ada_acc else []
        uni_accs = [r["accuracy"] for r in uni_acc] if uni_acc else []
        ada_m, ada_s = _mean_std(ada_scores)
        uni_m, uni_s = _mean_std(uni_scores)
        ada_am, ada_as = _mean_std(ada_accs)
        uni_am, uni_as = _mean_std(uni_accs)
        ada_dec = f"{sum(int(bool(d['decision'])) for d in ada_data)}/{len(ada_data)}"
        uni_dec = f"{sum(int(bool(d['decision'])) for d in uni_data)}/{len(uni_data)}"
        rows.append([ds_label, "Adaptive",
                      f"{ada_m:.4f} ± {ada_s:.4f}", ada_dec,
                      f"{ada_am:.1%} ± {ada_as:.1%}" if ada_accs else "N/A",
                      ", ".join(f"{s:.4f}" for s in ada_scores)])
        rows.append([ds_label, "Uniform (off)",
                      f"{uni_m:.4f} ± {uni_s:.4f}", uni_dec,
                      f"{uni_am:.1%} ± {uni_as:.1%}" if uni_accs else "N/A",
                      ", ".join(f"{s:.4f}" for s in uni_scores)])
    save_table(rows, "table-ablation.csv",
               ["Dataset", "Allocation", "Owner Score (mean±std)", "Pass Rate",
                "Accuracy (mean±std)", "Per-seed Scores"])


def _table_attack_robustness():
    attack_types = ["finetune", "quantization"]
    attack_labels = ["Finetune", "Quantization"]

    rows = []
    for ds_prefix, ds_label, ds_path, alloc in [
        ("cifar10-robustness", "CIFAR-10", os.path.join(RUNS_DIR, "cifar10-main-adaptive-tuneB"), "Adaptive"),
        ("cifar10-ablation-off-robustness", "CIFAR-10", os.path.join(RUNS_DIR, "cifar10-ablation-off-tuneB"), "Uniform"),
        ("mnist-robustness", "MNIST", os.path.join(RUNS_DIR, "mnist-main-adaptive"), "Adaptive"),
    ]:
        pre_data = _read_run_verifications(ds_path)
        pre_mean = np.mean([d["owner_score"] for d in pre_data])

        for atk, label in zip(attack_types, attack_labels):
            scores_all = []
            decisions_all = []
            for seed in [0, 1, 2]:
                import glob as _glob
                pattern = os.path.join(ATTACKS_DIR, f"{ds_prefix}-seed{seed}",
                                       f"{atk}-*", "verification_after_attack.json")
                matches = _glob.glob(pattern)
                if matches:
                    with open(matches[0]) as f:
                        d = json.load(f)
                    scores_all.append(d["owner_score"])
                    decisions_all.append(d["decision"])
            if not scores_all:
                continue
            m, s = _mean_std(scores_all)
            pass_rate = f"{sum(int(bool(d)) for d in decisions_all)}/{len(decisions_all)}"
            delta = m - pre_mean
            rows.append([f"{ds_label} ({alloc})", label, f"{pre_mean:.4f}",
                          f"{m:.4f} +/- {s:.4f}", f"{delta:+.4f}", pass_rate,
                          ", ".join(f"{v:.4f}" for v in scores_all)])

    save_table(rows, "table-attack-robustness.csv",
               ["Dataset (Alloc)", "Attack", "Pre-attack Score", "Post-attack Score (mean+/-std)",
                "Delta Score", "Pass Rate", "Per-seed Scores"])


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 – Utility–ambiguity tradeoff (MNIST adaptive vs uniform)
# ══════════════════════════════════════════════════════════════════════════════

def _read_accuracy_json(run_dir):
    """Read accuracy.json if present, return list of {seed, accuracy, ...}."""
    results = []
    for sd in sorted(d for d in os.listdir(run_dir)
                      if os.path.isdir(os.path.join(run_dir, d))):
        path = os.path.join(run_dir, sd, "accuracy.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return results


def plot_fig7_utility_tradeoff():
    ds_configs = [
        ("CIFAR-10", os.path.join(RUNS_DIR, "cifar10-main-adaptive-tuneB"),
         "cifar10", True),
        ("MNIST", os.path.join(RUNS_DIR, "mnist-main-adaptive"),
         "mnist", True),
    ]
    # add uniform/ablation counterparts
    ablation_configs = [
        ("CIFAR-10", os.path.join(RUNS_DIR, "cifar10-ablation-off-tuneB"),
         "cifar10", False),
        ("MNIST", os.path.join(RUNS_DIR, "mnist-ablation-off"),
         "mnist", False),
    ]

    names, ada_acc, uni_acc = [], [], []
    ada_owner = []; uni_owner = []
    ada_neg = []; uni_neg = []
    ada_pass = []; uni_pass = []

    for i, (ds_list, label) in enumerate([(ds_configs, "Adaptive"), (ablation_configs, "Uniform")]):
        for name, path, key, adaptive in ds_list:
            vdata = _read_run_verifications(path)
            adata = _read_accuracy_json(path)
            if not vdata or not adata:
                continue
            accs = [r["accuracy"] for r in adata]
            owners = [d["owner_score"] for d in vdata]
            negs = [d.get("negative_asr", 0) for d in vdata]
            decisions = [d["decision"] for d in vdata]

            if adaptive:
                names.append(name)
                ada_acc.append((np.mean(accs), np.std(accs, ddof=1)))
                ada_owner.append((np.mean(owners), np.std(owners, ddof=1)))
                ada_neg.append((np.mean(negs), np.std(negs, ddof=1)))
                ada_pass.append(f"{sum(int(bool(d)) for d in decisions)}/{len(decisions)}")
            else:
                uni_acc.append((np.mean(accs), np.std(accs, ddof=1)))
                uni_owner.append((np.mean(owners), np.std(owners, ddof=1)))
                uni_neg.append((np.mean(negs), np.std(negs, ddof=1)))
                uni_pass.append(f"{sum(int(bool(d)) for d in decisions)}/{len(decisions)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))
    x = np.arange(len(names))
    width = 0.30

    # (a) Task accuracy
    bars_a = ax1.bar(x - width / 2, [m for m, _ in ada_acc], width, yerr=[s for _, s in ada_acc],
                      capsize=6, color=COLORS["adaptive"], alpha=0.9,
                      edgecolor="white", linewidth=0.8, error_kw={"lw": 1.2}, label="Adaptive")
    bars_u = ax1.bar(x + width / 2, [m for m, _ in uni_acc], width, yerr=[s for _, s in uni_acc],
                      capsize=6, color=COLORS["uniform"], alpha=0.7,
                      edgecolor="white", linewidth=0.8, error_kw={"lw": 1.2}, label="Uniform")

    for bar, (m, _) in zip(bars_a, ada_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{m:.1%}", ha="center", fontsize=8, fontweight="bold", color=COLORS["adaptive"])
    for bar, (m, _) in zip(bars_u, uni_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{m:.1%}", ha="center", fontsize=8, fontweight="bold", color=COLORS["uniform"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Test Accuracy", fontsize=10)
    ax1.set_ylim(0, max(1.1, max(m for m, _ in ada_acc + uni_acc) + 0.2))
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_title("(a) Task Utility", fontweight="bold", fontsize=10)

    # (b) Owner score
    bars_oa = ax2.bar(x - width / 2, [m for m, _ in ada_owner], width, yerr=[s for _, s in ada_owner],
                       capsize=6, color=COLORS["adaptive"], alpha=0.9,
                       edgecolor="white", linewidth=0.8, error_kw={"lw": 1.2}, label="Adaptive")
    bars_ou = ax2.bar(x + width / 2, [m for m, _ in uni_owner], width, yerr=[s for _, s in uni_owner],
                       capsize=6, color=COLORS["uniform"], alpha=0.7,
                       edgecolor="white", linewidth=0.8, error_kw={"lw": 1.2}, label="Uniform")

    ax2.axhline(y=THRESHOLD, color=COLORS["threshold"], linestyle="--", linewidth=1.5,
                 label=f"Threshold ({THRESHOLD})")

    for i, (bar, pr) in enumerate(zip(bars_oa, ada_pass)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 pr, ha="center", fontsize=8.5, fontweight="bold", color=COLORS["pass"])
    for i, (bar, pr) in enumerate(zip(bars_ou, uni_pass)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 pr, ha="center", fontsize=8.5, fontweight="bold", color=COLORS["pass"])

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("Owner Score", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_title("(b) Verification (Owner Score)", fontweight="bold", fontsize=10)

    fig.suptitle("Utility–Verification Tradeoff: Adaptive vs Uniform Allocation",
                 fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure-07-utility-tradeoff")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 – FPR evaluation
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig8_fpr():
    report_st = os.path.join(RUNS_DIR, "cifar10-fpr-nonowners",
                             "fpr_evaluation_report_single-trigger.json")
    multicb_dir = os.path.join(RUNS_DIR, "cifar10-fpr-nonowners",
                               "reports_multicodebook")

    owner_data_file = os.path.join(RUNS_DIR, "cifar10-hadamard")
    owner_data = _read_run_verifications(owner_data_file)
    owner_scores = [d["owner_score"] for d in owner_data] if owner_data else []
    owner_mean = np.mean(owner_scores) if owner_scores else 0.73

    # Pool multi-codebook results
    clean_mb_all = []
    total_fp = 0
    total_n = 0
    if os.path.isdir(multicb_dir):
        for p in sorted(Path(multicb_dir).glob("fpr_codebook_seed*.json")):
            rmb = json.loads(p.read_text())
            scores = [r["owner_score"] for r in rmb.get("per_seed_results", [])]
            clean_mb_all.extend(scores)
            total_fp += sum(1 for r in rmb.get("per_seed_results", []) if r.get("passed"))
            total_n += len(rmb.get("per_seed_results", []))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bins = np.linspace(-0.1, 1.05, 32)

    # Left: Multi-bit FPR histogram (pooled over 10 codebooks)
    if clean_mb_all:
        ax1.hist(clean_mb_all, bins=bins, color=COLORS["competitor"], alpha=0.6,
                 edgecolor="gray", linewidth=0.8,
                 label=f"Clean models (n={len(clean_mb_all)})")
        ax1.set_title(f"(a) Multi-bit Pooled FPR = {total_fp}/{total_n} ({total_fp/total_n:.3f})",
                      fontweight="bold", fontsize=10)
    else:
        ax1.set_title("(a) Multi-bit (reports missing)", fontweight="bold", fontsize=10)

    for s in owner_scores:
        ax1.axvline(x=s, color=COLORS["owner"], linestyle="--", alpha=0.5, linewidth=1.2)
    ax1.axvline(x=owner_mean, color=COLORS["owner"], linestyle="-", linewidth=2.0)
    ax1.axvline(x=THRESHOLD, color=COLORS["threshold"], linestyle="-", linewidth=1.5,
                label=f"$\\tau={THRESHOLD}$")
    ax1.set_xlabel("Owner Score $s_i$", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.set_xlim(-0.1, 1.05)
    ax1.legend(fontsize=7.5, loc="upper left")

    # Right: Single-trigger FPR histogram
    if os.path.exists(report_st):
        with open(report_st) as f:
            rst = json.load(f)
        clean_st = [r["owner_score"] for r in rst.get("per_seed_results", [])]
        npass = sum(1 for r in rst.get("per_seed_results", []) if r.get("passed"))
        ax2.hist(clean_st, bins=bins, color=COLORS["fail"], alpha=0.6,
                 edgecolor="gray", linewidth=0.8,
                 label=f"Clean models (n={len(clean_st)})")
        ax2.set_title(f"(b) Single-trigger Codebook FPR = {npass}/{len(clean_st)}",
                      fontweight="bold", fontsize=10)
    else:
        ax2.set_title("(b) Single-trigger (report missing)", fontweight="bold", fontsize=10)

    for s in owner_scores:
        ax2.axvline(x=s, color=COLORS["owner"], linestyle="--", alpha=0.3, linewidth=1.2)
    ax2.axvline(x=THRESHOLD, color=COLORS["threshold"], linestyle="-", linewidth=1.5)
    ax2.set_xlabel("Owner Score $s_i$", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_xlim(-0.1, 1.05)
    ax2.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Empirical FPR Validation: Multi-bit vs Single-trigger (CIFAR-10, $m=64$, 10 codebooks $\\times$ 20 clean models)",
                 fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure-08-fpr-evaluation")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== Figures ===")
    print("  Fig 1 – Method framework...")
    plot_fig1_framework()
    print("  Fig 2 – Main results...")
    plot_fig2_main_results()
    print("  Fig 3 – Attack robustness...")
    plot_fig3_attack_robustness()
    print("  Fig 4 – Non-IID effects...")
    plot_fig4_non_iid()
    print("  Fig 5 – Competitor distribution...")
    plot_fig5_competitor_distribution()
    print("  Fig 6 – Training convergence...")
    plot_fig6_training_convergence()
    print("  Fig 7 – Utility–verification tradeoff...")
    plot_fig7_utility_tradeoff()
    print("  Fig 8 – FPR evaluation...")
    plot_fig8_fpr()

    print("\n=== Tables ===")
    print("  Table – Main results...")
    _table_main_results()
    print("  Table – Ablation...")
    _table_ablation()
    print("  Table – Attack robustness (adaptive + ablation)...")
    _table_attack_robustness()

    print(f"\nDone! Output to {FIG_DIR}/ and {TABLE_DIR}/")


if __name__ == "__main__":
    main()
