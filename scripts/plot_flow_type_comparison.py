"""3-way head-to-head: NSF vs FMPE vs Mixture-NSF on the same (task, seed).

Reads:
  c2st_decomp/{task}_s{seed}{,_fmpe,_mixture_nsf_K2}.npz
  mode_count/{task}_s{seed}{,_fmpe,_mixture_nsf_K2}.npz
  flow_vs_quantile/{task}_s{seed}{,_fmpe,_mixture_nsf_K2}.npz   (for pinball)

Default scope: two_moons_distractors s42 (the mode-collapse case where the
diagnostic predicted mixture would help and FMPE wouldn't).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
MODE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/mode_count")
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

FLOW_LABELS = [
    ("nsf", "NSF (default)", "C0"),
    ("fmpe", "FMPE", "C3"),
    ("mixture_nsf_K2", "Mixture-NSF (K=2)", "C2"),
]


def load_for(task: str, seed: int) -> list[dict]:
    rows = []
    for tag, label, color in FLOW_LABELS:
        suffix = "" if tag == "nsf" else f"_{tag}"
        d_decomp = DECOMP_DIR / f"{task}_s{seed}{suffix}.npz"
        d_mode = MODE_DIR / f"{task}_s{seed}{suffix}.npz"
        d_sample = SAMPLE_DIR / f"{task}_s{seed}{suffix}.npz"
        if not (d_decomp.exists() and d_mode.exists() and d_sample.exists()):
            print(f"  [skip] {tag} (missing files)")
            continue
        decomp = dict(np.load(d_decomp, allow_pickle=True))
        mode = dict(np.load(d_mode, allow_pickle=True))
        sample = dict(np.load(d_sample, allow_pickle=True))
        rows.append({
            "tag": tag, "label": label, "color": color,
            "joint": decomp["joint"], "marginal": decomp["marginal"], "rank": decomp["rank"],
            "flow_count": mode["flow_count"], "mcmc_count": mode["mcmc_count"],
            "n_match": int(mode["n_match"]), "n_collapse": int(mode["n_collapse"]),
            "n_overcount": int(mode["n_overcount"]),
            "flow_pinball": float(sample["flow_pinball"]),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="two_moons_distractors")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_for(args.task, args.seed)
    if not rows:
        print("No flow-type results found.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── A: C2ST decomposition bars per flow type ──
    ax = axes[0]
    metrics = ["joint", "marginal", "rank"]
    metric_labels = ["joint", "marginal-only", "rank (copula)"]
    xs = np.arange(len(metrics))
    w = 0.27
    for i, r in enumerate(rows):
        offs = (i - (len(rows) - 1) / 2) * w
        means = [r[m].mean() for m in metrics]
        stds = [r[m].std() for m in metrics]
        ax.bar(xs + offs, means, w, yerr=stds, color=r["color"],
               label=r["label"], capsize=3)
    ax.axhline(0.5, color="grey", ls=":", alpha=0.7, label="C2ST = 0.5")
    ax.set_xticks(xs); ax.set_xticklabels(metric_labels)
    ax.set_ylabel("C2ST (lower = better)")
    n_obs = len(rows[0]["joint"])
    ax.set_title(f"A. C2ST decomposition (n={n_obs} ref obs)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # ── B: mode-count agreement per flow type ──
    ax = axes[1]
    n_total = len(rows[0]["flow_count"])
    cats = ["match", "collapse", "over"]
    xs = np.arange(len(cats))
    for i, r in enumerate(rows):
        offs = (i - (len(rows) - 1) / 2) * w
        vals = [r["n_match"] / n_total, r["n_collapse"] / n_total,
                r["n_overcount"] / n_total]
        ax.bar(xs + offs, vals, w, color=r["color"], label=r["label"])
    ax.set_xticks(xs); ax.set_xticklabels(["match", "flow < MCMC\n(collapse)",
                                            "flow > MCMC\n(over-count)"])
    ax.set_ylabel(f"Fraction of {n_total} ref obs")
    ax.set_title(f"B. Mode-count agreement (n={n_total} obs)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── C: pinball + summary table panel ──
    ax = axes[2]
    pin_xs = np.arange(len(rows))
    pinballs = [r["flow_pinball"] for r in rows]
    ax.bar(pin_xs, pinballs, color=[r["color"] for r in rows])
    ax.set_xticks(pin_xs); ax.set_xticklabels([r["label"] for r in rows],
                                              rotation=15, ha="right")
    ax.set_ylabel("Flow pinball (true θ on ref obs)")
    ax.set_title("C. Marginal pinball")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(pinballs):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    fig.suptitle(
        f"{args.task} | seed={args.seed} — head-to-head: NSF vs FMPE vs Mixture-NSF",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / f"flow_type_comparison_{args.task}_s{args.seed}.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")

    print("\n=== Summary ===")
    print(f"{'flow_type':<22} {'joint':>8} {'marg':>8} {'rank':>8} "
          f"{'pinball':>9} {'match':>6} {'collapse':>9}")
    for r in rows:
        print(f"{r['label']:<22} {r['joint'].mean():>8.4f} "
              f"{r['marginal'].mean():>8.4f} {r['rank'].mean():>8.4f} "
              f"{r['flow_pinball']:>9.4f} "
              f"{r['n_match']:>6} {r['n_collapse']:>9}")


if __name__ == "__main__":
    main()
