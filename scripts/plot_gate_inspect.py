"""Inspect what the gating network learns at each K.

For each (task, seed, K), we have gate softmax outputs π on train and ref
observations (saved by train_and_sample_flow.py). Two questions:

  Sparsifying — does the gate concentrate on a subset of components?
    Read: max(π) per observation. Sparse → max π → 1.0.

  Component coverage — are some components dead?
    Read: per-component max π over the train sample. Dead component →
    max π near 0.

The "plateau is free" hypothesis predicts that as K grows beyond
K_required, max π → 1.0 with most components dead. If instead max π
stays near 1/K (uniform), the gate is balancing rather than selecting,
and large-K mixture is over-parameterized.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SAMPLE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="two_moons_distractors")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K-list", type=int, nargs="+", default=[2, 4, 8])
    args = ap.parse_args()

    rows = []
    for K in args.K_list:
        p = SAMPLE_DIR / f"{args.task}_s{args.seed}_mixture_nsf_K{K}.npz"
        if not p.exists():
            print(f"  [skip] K={K} (missing {p})")
            continue
        d = np.load(p, allow_pickle=True)
        if "gate_pi_train" not in d.files:
            print(f"  [skip] K={K} (no gate diagnostic stored)")
            continue
        rows.append({
            "K": K,
            "pi_train": d["gate_pi_train"],         # (n_train_sub, K)
            "pi_ref": d["gate_pi_ref"],             # (n_ref, K)
        })
    if not rows:
        print("No gate diagnostic data found.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── A: histogram of max(π) per train observation, one curve per K ──
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(rows)))
    for color, r in zip(cmap, rows, strict=False):
        max_pi = r["pi_train"].max(axis=-1)        # (n_train_sub,)
        ax.hist(max_pi, bins=40, range=(1.0 / r["K"], 1.0),
                histtype="step", lw=2, color=color,
                label=f"K={r['K']} (max π mean={max_pi.mean():.3f})",
                density=True)
        ax.axvline(1.0 / r["K"], color=color, ls=":", alpha=0.5)
    ax.set_xlabel("max π over components, per train obs")
    ax.set_ylabel("Density")
    ax.set_title("A. Gate concentration on train data")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── B: entropy of π per train obs, one curve per K ──
    ax = axes[1]
    for color, r in zip(cmap, rows, strict=False):
        H = -np.sum(r["pi_train"] * np.log(r["pi_train"] + 1e-12), axis=-1)
        ax.hist(H, bins=40, histtype="step", lw=2, color=color,
                label=(f"K={r['K']} (mean H={H.mean():.3f}, "
                       f"max log K={float(np.log(r['K'])):.3f})"),
                density=True)
        ax.axvline(float(np.log(r["K"])), color=color, ls=":", alpha=0.5)
    ax.set_xlabel("Entropy H(π), per train obs")
    ax.set_ylabel("Density")
    ax.set_title("B. Gate entropy (0 = sparse, log K = uniform)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── C: per-component max π over train (component liveness) ──
    ax = axes[2]
    width = 0.85 / len(rows)
    for i, r in enumerate(rows):
        per_comp_max = r["pi_train"].max(axis=0)    # (K,)
        per_comp_mean = r["pi_train"].mean(axis=0)
        xs = np.arange(r["K"]) + (i - (len(rows) - 1) / 2) * width
        ax.bar(xs, per_comp_max, width=width, color=cmap[i],
               alpha=0.85, label=f"K={r['K']} max")
        ax.scatter(xs, per_comp_mean, color="black", s=15, zorder=4,
                   label=f"K={r['K']} mean" if i == 0 else None)
    ax.axhline(0.5, color="grey", ls=":", alpha=0.5)
    ax.set_xlabel("Component index")
    ax.set_ylabel("max / mean π over train")
    ax.set_title("C. Per-component liveness")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Gate inspection: {args.task} | seed={args.seed} (deterministic)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / f"gate_inspect_{args.task}_s{args.seed}.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")

    print("\n=== Gate summary ===")
    print(f"{'K':>3} {'max π mean':>11} {'max π std':>11} {'H mean':>9} "
          f"{'log K':>9} {'dead components (max π < 0.05)':>35}")
    for r in rows:
        max_pi = r["pi_train"].max(axis=-1)
        H = -np.sum(r["pi_train"] * np.log(r["pi_train"] + 1e-12), axis=-1)
        per_comp_max = r["pi_train"].max(axis=0)
        n_dead = int((per_comp_max < 0.05).sum())
        print(f"{r['K']:>3} {max_pi.mean():>11.3f} {max_pi.std():>11.3f} "
              f"{H.mean():>9.3f} {float(np.log(r['K'])):>9.3f} {n_dead:>35d}")


if __name__ == "__main__":
    main()
