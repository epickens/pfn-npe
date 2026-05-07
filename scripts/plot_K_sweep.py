"""K-sweep summary: how do C2ST and mode-count metrics behave as K grows?

Plots joint/marginal/rank C2ST and mode-match fraction vs K for a single
(task, seed). NSF and FMPE values are shown as horizontal reference lines
on the C2ST panel for context.

The hypothesis we are testing: gating-network capacity makes large K free.
If the curves plateau at K ≥ K_required (rather than peak at K_required and
fall off), then "default K = 8" is a deployable recipe.
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


def load_one(task: str, seed: int, suffix: str) -> dict | None:
    d_decomp = DECOMP_DIR / f"{task}_s{seed}{suffix}.npz"
    d_mode = MODE_DIR / f"{task}_s{seed}{suffix}.npz"
    d_sample = SAMPLE_DIR / f"{task}_s{seed}{suffix}.npz"
    if not (d_decomp.exists() and d_sample.exists()):
        return None
    decomp = dict(np.load(d_decomp, allow_pickle=True))
    sample = dict(np.load(d_sample, allow_pickle=True))
    out = {
        "joint": float(decomp["joint"].mean()),
        "marginal": float(decomp["marginal"].mean()),
        "rank": float(decomp["rank"].mean()),
        "joint_std": float(decomp["joint"].std()),
        "rank_std": float(decomp["rank"].std()),
        "pinball": float(sample.get("flow_pinball", float("nan"))),
    }
    if d_mode.exists():
        mode = dict(np.load(d_mode, allow_pickle=True))
        out.update({
            "n_match": int(mode["n_match"]),
            "n_collapse": int(mode["n_collapse"]),
            "n_total": len(mode["flow_count"]),
        })
    else:
        out.update({"n_match": 0, "n_collapse": 0, "n_total": 1})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="two_moons_distractors")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K-list", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()

    results: list[tuple[int, dict]] = []
    for K in args.K_list:
        r = load_one(args.task, args.seed, f"_mixture_nsf_K{K}")
        if r is not None:
            results.append((K, r))
    if not results:
        print("No mixture-NSF K-sweep results found.")
        return

    nsf = load_one(args.task, args.seed, "")
    fmpe = load_one(args.task, args.seed, "_fmpe")

    Ks = np.array([k for k, _ in results])
    joints = np.array([r["joint"] for _, r in results])
    margs = np.array([r["marginal"] for _, r in results])
    ranks = np.array([r["rank"] for _, r in results])
    joint_stds = np.array([r["joint_std"] for _, r in results])
    rank_stds = np.array([r["rank_std"] for _, r in results])
    pinballs = np.array([r["pinball"] for _, r in results])
    matches = np.array([r["n_match"] / r["n_total"] for _, r in results])
    collapses = np.array([r["n_collapse"] / r["n_total"] for _, r in results])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── A: C2ST decomposition vs K ──
    ax = axes[0]
    ax.errorbar(Ks, joints, yerr=joint_stds, marker="o", color="C0",
                label="joint", lw=2, capsize=3)
    ax.plot(Ks, margs, marker="s", color="C2", label="marginal", lw=2)
    ax.errorbar(Ks, ranks, yerr=rank_stds, marker="^", color="C3",
                label="rank (copula)", lw=2, capsize=3)
    if nsf is not None:
        ax.axhline(nsf["joint"], color="C0", ls="--", alpha=0.5,
                   label=f"NSF joint ({nsf['joint']:.3f})")
        ax.axhline(nsf["rank"], color="C3", ls="--", alpha=0.5,
                   label=f"NSF rank ({nsf['rank']:.3f})")
    if fmpe is not None:
        ax.axhline(fmpe["joint"], color="grey", ls=":", alpha=0.7,
                   label=f"FMPE joint ({fmpe['joint']:.3f})")
    ax.axhline(0.5, color="black", ls=":", alpha=0.5)
    ax.set_xlabel("K (mixture components)")
    ax.set_ylabel("C2ST (lower = better)")
    ax.set_xticks(Ks)
    ax.set_title("A. C2ST decomposition vs K")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # ── B: pinball + mode-match vs K, twin axes ──
    ax1 = axes[1]
    color1 = "C4"
    ax1.plot(Ks, pinballs, "o-", color=color1, lw=2, label="pinball")
    ax1.set_xlabel("K (mixture components)")
    ax1.set_ylabel("Pinball loss", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(Ks); ax1.grid(True, alpha=0.3)
    ax1.set_title("B. Pinball + mode-count match vs K")

    if nsf is not None:
        ax1.axhline(nsf["pinball"], color=color1, ls="--", alpha=0.4,
                    label=f"NSF pinball ({nsf['pinball']:.3f})")

    ax2 = ax1.twinx()
    color2 = "C5"
    ax2.plot(Ks, matches, "s--", color=color2, lw=2, label="mode match")
    ax2.plot(Ks, collapses, "^:", color="C3", lw=1.5, alpha=0.7,
             label="mode collapse")
    ax2.set_ylabel("Fraction of ref obs", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-0.05, 1.05)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")

    fig.suptitle(
        f"K sweep: mixture-NSF on {args.task} | seed={args.seed}",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / f"K_sweep_{args.task}_s{args.seed}.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")

    print("\n=== K-sweep summary ===")
    print(f"{'K':>3} {'joint':>9} {'marg':>9} {'rank':>9} {'pinball':>9} "
          f"{'match':>8} {'collapse':>10}")
    for K, r in results:
        print(f"{K:>3} {r['joint']:>9.4f} {r['marginal']:>9.4f} "
              f"{r['rank']:>9.4f} {r['pinball']:>9.4f} "
              f"{r['n_match']/r['n_total']:>8.2f} "
              f"{r['n_collapse']/r['n_total']:>10.2f}")
    if nsf:
        print(f"NSF: joint={nsf['joint']:.4f}, marg={nsf['marginal']:.4f}, "
              f"rank={nsf['rank']:.4f}, pinball={nsf['pinball']:.4f}, "
              f"match={nsf['n_match']/nsf['n_total']:.2f}")
    if fmpe:
        print(f"FMPE: joint={fmpe['joint']:.4f}, marg={fmpe['marginal']:.4f}, "
              f"rank={fmpe['rank']:.4f}, pinball={fmpe['pinball']:.4f}, "
              f"match={fmpe['n_match']/fmpe['n_total']:.2f}")


if __name__ == "__main__":
    main()
