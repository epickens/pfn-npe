"""Cross-task summary of the C2ST decomposition (joint / marginal / rank).

Each (task, seed) contributes 10 ref-obs C2ST scores at three levels:
  joint     = vanilla C2ST in θ-space (the standard benchmark metric)
  marginal  = mean of per-dim 1D C2ST (marginal-only mismatch)
  rank      = C2ST after pooled-empirical-CDF rank transform per dim
              (marginals exactly matched; remaining gap is copula)

If rank C2ST stays high while marginal C2ST sits at 0.5, the joint failure
is in the dependence structure between θ-dims, not in any 1D marginal.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
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

TASK_ORDER = [
    ("slcp", "slcp"),
    ("two_moons_distractors", "two_moons+distr"),
    ("gaussian_mixture_distractors", "gaussian_mixture+distr"),
    ("bernoulli_glm_distractors", "bernoulli_glm+distr"),
    ("sir_distractors", "sir+distr"),
]


def main() -> None:
    rows = []
    for task, label in TASK_ORDER:
        for p in sorted(DECOMP_DIR.glob(f"{task}_s*.npz")):
            d = dict(np.load(p, allow_pickle=True))
            rows.append({
                "task": task, "label": label, "seed": int(d["seed"]),
                "joint": d["joint"], "marginal": d["marginal"], "rank": d["rank"],
                "marginal_per_dim": d["marginal_per_dim"],
            })
    if not rows:
        print("No decomposition results found.")
        return

    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    task_names = [t for t, _ in TASK_ORDER if t in by_task]
    task_labels = {t: lab for t, lab in TASK_ORDER}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── A: bars per task for joint/marginal/rank, seed-pooled ──
    ax = axes[0]
    xs = np.arange(len(task_names))
    w = 0.27
    for kind, off, color in [("joint", -w, "C0"),
                             ("marginal", 0.0, "C2"),
                             ("rank", w, "C3")]:
        means = []
        stds = []
        for t in task_names:
            vals = np.concatenate([r[kind] for r in by_task[t]])
            means.append(vals.mean())
            stds.append(vals.std())
        ax.bar(xs + off, means, w, yerr=stds, color=color,
               label=kind, capsize=3)
    ax.axhline(0.5, color="grey", ls=":", alpha=0.7, label="C2ST = 0.5 (ideal)")
    ax.set_xticks(xs); ax.set_xticklabels([task_labels[t] for t in task_names],
                                          rotation=20, ha="right")
    ax.set_ylabel("C2ST")
    n_total = sum(len(r["joint"]) for r in rows)
    ax.set_title(
        f"A. C2ST decomposition per task (n={n_total} total ref-obs scores)"
    )
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # ── B: joint vs rank scatter per ref obs ──
    ax = axes[1]
    cmap = plt.cm.tab10(np.linspace(0, 1, len(task_names)))
    for color, t in zip(cmap, task_names, strict=False):
        joint_vals = np.concatenate([r["joint"] for r in by_task[t]])
        rank_vals = np.concatenate([r["rank"] for r in by_task[t]])
        ax.scatter(joint_vals, rank_vals, color=color, s=35,
                   edgecolor="black", lw=0.4, alpha=0.7,
                   label=task_labels[t])
    lo = 0.45; hi = 1.0
    ax.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.6, label="y=x")
    ax.set_xlabel("Joint C2ST (θ-space)")
    ax.set_ylabel("Rank C2ST (copula)")
    ax.set_title("B. Rank vs joint per ref obs")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # ── C: stacked decomposition: above-0.5 = marginal_part + (rank-marg) + (joint-rank) ──
    ax = axes[2]
    width = 0.65
    bottom = 0.5
    for i, t in enumerate(task_names):
        joint = np.concatenate([r["joint"] for r in by_task[t]]).mean()
        marg = np.concatenate([r["marginal"] for r in by_task[t]]).mean()
        rank = np.concatenate([r["rank"] for r in by_task[t]]).mean()
        marg_part = max(marg - 0.5, 0.0)
        rank_extra = max(rank - marg, 0.0)
        joint_minus_rank = joint - rank
        # Stack: marginals + (rank − marg) + (joint − rank). Negative residuals
        # plotted in grey on top so the bar height equals joint.
        ax.bar(i, marg_part, width, bottom=bottom, color="C2",
               label="marginal contribution" if i == 0 else None)
        ax.bar(i, rank_extra, width, bottom=bottom + marg_part, color="C3",
               label="copula contribution (rank − marginal)" if i == 0 else None)
        if joint_minus_rank > 0:
            ax.bar(i, joint_minus_rank, width,
                   bottom=bottom + marg_part + rank_extra, color="C0",
                   label="joint − rank residual" if i == 0 else None)
        else:
            ax.bar(i, joint_minus_rank, width, bottom=rank,
                   color="lightgrey",
                   label="joint < rank (rank inflated)" if i == 0 else None)
        ax.scatter(i, joint, s=40, color="black", zorder=4)
    ax.axhline(0.5, color="grey", ls=":", alpha=0.7)
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels([task_labels[t] for t in task_names],
                       rotation=20, ha="right")
    ax.set_ylabel("C2ST decomposition (above 0.5)")
    ax.set_title(
        "C. Stack: 0.5 + marginal + copula + residual; ● = joint mean"
    )
    ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Trained NSF flow vs MCMC: marginal vs copula failure decomposition",
        fontsize=11,
    )
    fig.tight_layout()
    out_full = FIG_DIR / "c2st_decomposition_summary_full.png"
    fig.savefig(str(out_full), bbox_inches="tight")
    print(f"Wrote {out_full}")

    # ── Simple §5.3 figure: joint vs marginal bars only (no rank) ──
    fig_s, ax_s = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(task_names))
    w = 0.38
    for kind, off, color in [("joint", -w / 2, "C0"),
                             ("marginal", w / 2, "C2")]:
        means = []
        stds = []
        for t in task_names:
            vals = np.concatenate([r[kind] for r in by_task[t]])
            means.append(vals.mean())
            stds.append(vals.std())
        ax_s.bar(xs + off, means, w, yerr=stds, color=color,
                 label=kind, capsize=3)
    ax_s.axhline(0.5, color="grey", ls=":", alpha=0.7,
                 label="C2ST = 0.5 (ideal)")
    ax_s.set_xticks(xs)
    ax_s.set_xticklabels([task_labels[t] for t in task_names],
                         rotation=20, ha="right")
    ax_s.set_ylabel("C2ST")
    ax_s.set_title(
        f"Joint vs marginal C2ST per task (n={n_total} ref-obs scores)"
    )
    ax_s.legend()
    ax_s.grid(True, axis="y", alpha=0.3)
    fig_s.tight_layout()
    out_simple = FIG_DIR / "c2st_decomposition_summary.png"
    fig_s.savefig(str(out_simple), bbox_inches="tight")
    print(f"Wrote {out_simple}")

    print("\n=== Per-task means ===")
    print(f"{'task':<32} {'n_obs':>6} {'joint':>9} {'marg':>9} {'rank':>9} "
          f"{'rank-marg':>10} {'joint-marg':>11}")
    for t in task_names:
        joint = np.concatenate([r["joint"] for r in by_task[t]])
        marg = np.concatenate([r["marginal"] for r in by_task[t]])
        rank = np.concatenate([r["rank"] for r in by_task[t]])
        print(f"{t:<32} {len(joint):>6} {joint.mean():>9.4f} "
              f"{marg.mean():>9.4f} {rank.mean():>9.4f} "
              f"{(rank - marg).mean():>+10.4f} {(joint - marg).mean():>+11.4f}")


if __name__ == "__main__":
    main()
