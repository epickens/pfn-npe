"""Cross-task summary of the mode-count diagnostic.

Aggregates per-(task, seed) mode counts (10 ref obs each) for 2D-θ tasks
and produces a 2-panel figure:
  A. Per-task bars: fraction of obs (across seeds) where flow matches /
     under-counts (collapse) / over-counts MCMC modes.
  B. Scatter of flow vs MCMC mode counts across (task, seed, obs), with
     y=x diagonal. Mode collapse = points below the diagonal.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/mode_count")

# Accept `{task}_s{seed}.npz` (default = NSF) or `{task}_s{seed}_nsf.npz`
# (explicit --flow-type nsf). Excludes ablation variants like _fmpe,
# _mixture_nsf_K1, etc.
def _is_canonical(path: Path, task: str) -> bool:
    return bool(re.match(rf"^{re.escape(task)}_s\d+(_nsf)?$", path.stem))
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
    ("two_moons", "two_moons"),
    ("two_moons_distractors", "two_moons+distr"),
    ("gaussian_mixture", "gaussian_mixture"),
    ("gaussian_mixture_distractors", "gaussian_mixture+distr"),
    ("sir", "sir"),
    ("sir_distractors", "sir+distr"),
]


def main() -> None:
    rows = []
    for task, label in TASK_ORDER:
        for p in sorted(DECOMP_DIR.glob(f"{task}_s*.npz")):
            if not _is_canonical(p, task):
                continue
            d = dict(np.load(p, allow_pickle=True))
            rows.append({
                "task": task, "label": label, "seed": int(d["seed"]),
                "flow_count": d["flow_count"],
                "mcmc_count": d["mcmc_count"],
            })
    if not rows:
        print("No mode_count results found.")
        return

    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    task_names = [t for t, _ in TASK_ORDER if t in by_task]
    task_labels = {t: lab for t, lab in TASK_ORDER}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── A: bars of match / collapse / overcount fractions per task ──
    ax = axes[0]
    xs = np.arange(len(task_names))
    w = 0.27
    match_frac, collapse_frac, over_frac, totals = [], [], [], []
    for t in task_names:
        flow = np.concatenate([r["flow_count"] for r in by_task[t]])
        mcmc = np.concatenate([r["mcmc_count"] for r in by_task[t]])
        n = len(flow)
        match_frac.append((flow == mcmc).sum() / n)
        collapse_frac.append((flow < mcmc).sum() / n)
        over_frac.append((flow > mcmc).sum() / n)
        totals.append(n)
    ax.bar(xs - w, match_frac, w, color="C2", label="match")
    ax.bar(xs, collapse_frac, w, color="C3", label="flow < MCMC (collapse)")
    ax.bar(xs + w, over_frac, w, color="C0", label="flow > MCMC (over-count)")
    for i, n in enumerate(totals):
        ax.text(i, 1.02, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([task_labels[t] for t in task_names], rotation=20,
                       ha="right")
    ax.set_ylabel("Fraction of (seed × ref obs)")
    ax.set_title("A. Mode-count agreement per task")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    # ── B: scatter of flow vs MCMC mode count, jittered ──
    ax = axes[1]
    rng = np.random.default_rng(0)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(task_names)))
    for color, t in zip(cmap, task_names, strict=False):
        flow = np.concatenate([r["flow_count"] for r in by_task[t]])
        mcmc = np.concatenate([r["mcmc_count"] for r in by_task[t]])
        jx = rng.normal(0, 0.10, size=flow.size)
        jy = rng.normal(0, 0.10, size=flow.size)
        ax.scatter(mcmc + jx, flow + jy, color=color, s=45,
                   edgecolor="black", lw=0.4, alpha=0.7,
                   label=task_labels[t])
    lo = -0.5
    hi = max(np.concatenate([r["mcmc_count"] for r in rows]).max(),
             np.concatenate([r["flow_count"] for r in rows]).max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.6, label="y=x")
    ax.set_xlabel("MCMC mode count (jittered)")
    ax.set_ylabel("Flow mode count (jittered)")
    ax.set_title("B. Flow vs MCMC mode count per ref obs")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    fig.suptitle(
        f"Mode count: trained NSF flow vs MCMC ({sum(totals)} obs total)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "mode_count_summary.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")

    print("\n=== Per-task mode-count agreement ===")
    print(f"{'task':<32} {'n':>4} {'match':>8} {'collapse':>10} {'over':>8} "
          f"{'mean_flow':>10} {'mean_mcmc':>10}")
    for t in task_names:
        flow = np.concatenate([r["flow_count"] for r in by_task[t]])
        mcmc = np.concatenate([r["mcmc_count"] for r in by_task[t]])
        n = len(flow)
        print(f"{t:<32} {n:>4} {(flow == mcmc).sum():>8} "
              f"{(flow < mcmc).sum():>10} {(flow > mcmc).sum():>8} "
              f"{flow.mean():>10.2f} {mcmc.mean():>10.2f}")


if __name__ == "__main__":
    main()
