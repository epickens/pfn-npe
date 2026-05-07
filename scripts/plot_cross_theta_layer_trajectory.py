"""Per-layer trajectory of cross-θ diag and off-diag R² (paired).

§5.4 paper figure. Shows for each task family (distractor / standard SBIBM
/ high-dim time-series) the group-mean trajectory of diag R² (encoder
extracting θ_i from the i-th per-dim chunk) and off-diag R² (extracting
θ_j from the i-th chunk, i ≠ j). At L0 both are ≈ 0 (nothing encoded);
at L4 diag rises; whether off-diag tracks or stays low decides whether
the encoder is allocating per-dim subspaces.

Reads:
  - pfn_testing/sbi/outputs/layer_ablation/cross_theta/{task}_s{seed}.npz

Writes:
  - pfn_testing/sbi/outputs/layer_ablation/figures/cross_theta_layer_trajectory.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

CROSS_THETA_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/cross_theta")
FIGURES_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")

NAME_RE = re.compile(
    r"(?P<task>.+?)_s(?P<seed>\d+)(?P<extra>(_m\w+)?(_ls\w+)?)?\.npz$"
)

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

# Same family grouping as plot_probe_cross_task.py / plot_variance_probe_cross_task.py
DISTRACTOR_TASKS = {
    "two_moons_distractors", "gaussian_mixture_distractors",
    "sir_distractors", "slcp_distractors", "bernoulli_glm_distractors",
}
HIGHDIM_TASKS = {"ou", "solar_dynamo", "ar1_ts_t50", "lotka_volterra"}

GROUP_COLORS = {"distr": "C0", "standard": "C2", "highdim": "C3"}
GROUP_LABELS = {
    "distr": "distractor variants",
    "standard": "standard SBIBM",
    "highdim": "high-dim time-series",
}


def task_group(task: str) -> str:
    if task in DISTRACTOR_TASKS:
        return "distr"
    if task in HIGHDIM_TASKS:
        return "highdim"
    return "standard"


def load_long() -> pd.DataFrame:
    """One row per (task, seed, layer) with diag and off_diag means."""
    rows = []
    for npz_path in sorted(CROSS_THETA_DIR.glob("*.npz")):
        m = NAME_RE.match(npz_path.name)
        if m is None or m.group("extra"):
            continue
        d = np.load(npz_path, allow_pickle=True)
        layers = np.asarray(d["layers"])
        R2 = np.asarray(d["R2"])
        D = R2.shape[1]
        if D < 2:
            continue
        eye = np.eye(D, dtype=bool)
        for li, k in enumerate(layers):
            mat = R2[li]
            rows.append({
                "task": m.group("task"),
                "seed": int(m.group("seed")),
                "layer": int(k),
                "diag": float(np.mean(np.diag(mat))),
                "off_diag": float(np.nanmean(np.where(eye, np.nan, mat))),
                "D": D,
            })
    return pd.DataFrame(rows)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_long()
    if df.empty:
        raise SystemExit(f"No cross_theta npzs in {CROSS_THETA_DIR}")

    # Aggregate over seeds per (task, layer): seed-mean.
    agg = df.groupby(["task", "layer"]).agg(
        diag=("diag", "mean"),
        off_diag=("off_diag", "mean"),
    ).reset_index()

    # Pivot to {task: {"diag": (L,), "off_diag": (L,), "group": str}}.
    by_task: dict[str, dict] = {}
    for task, sub in agg.groupby("task"):
        sub = sub.sort_values("layer")
        by_task[task] = {
            "diag": sub["diag"].values,
            "off_diag": sub["off_diag"].values,
            "group": task_group(task),
        }

    # Group stats: mean and std across tasks within group.
    by_group: dict[str, list[dict]] = {"distr": [], "standard": [], "highdim": []}
    for task, d in by_task.items():
        by_group[d["group"]].append(d)
    group_stats = {}
    for g, tasks in by_group.items():
        if not tasks:
            continue
        diag_arr = np.stack([t["diag"] for t in tasks])
        off_arr = np.stack([t["off_diag"] for t in tasks])
        group_stats[g] = {
            "diag_mean": diag_arr.mean(0),
            "diag_std": diag_arr.std(0),
            "off_mean": off_arr.mean(0),
            "off_std": off_arr.std(0),
            "n_tasks": len(tasks),
        }

    layers = np.arange(12)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Per-task lines (faded background): diag solid, off-diag dashed.
    for task, d in by_task.items():
        color = GROUP_COLORS[d["group"]]
        ax.plot(layers, d["diag"], color=color, lw=0.6, alpha=0.20)
        ax.plot(layers, d["off_diag"], color=color, lw=0.6, alpha=0.20, ls="--")

    # Group-mean lines (foreground) + ±1 std band across tasks within group.
    for g in ("distr", "standard", "highdim"):
        if g not in group_stats:
            continue
        s = group_stats[g]
        color = GROUP_COLORS[g]
        # Diag = solid thick
        ax.plot(layers, s["diag_mean"], color=color, lw=2.6, alpha=0.95)
        ax.fill_between(
            layers, s["diag_mean"] - s["diag_std"], s["diag_mean"] + s["diag_std"],
            color=color, alpha=0.13, linewidth=0,
        )
        # Off-diag = dashed thick
        ax.plot(layers, s["off_mean"], color=color, lw=2.0, alpha=0.95, ls="--")
        ax.fill_between(
            layers, s["off_mean"] - s["off_std"], s["off_mean"] + s["off_std"],
            color=color, alpha=0.07, linewidth=0,
        )

    ax.axhline(0, color="grey", ls=":", lw=0.6, alpha=0.7)
    ax.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.4, zorder=0)

    # Two-tier legend: families + line-style key.
    family_handles = [
        mlines.Line2D([], [], color=GROUP_COLORS[g], lw=2.6,
                      label=f"{GROUP_LABELS[g]} (n={group_stats[g]['n_tasks']})")
        for g in ("distr", "standard", "highdim") if g in group_stats
    ]
    style_handles = [
        mlines.Line2D([], [], color="black", lw=2.0, label=r"diag $R^2$ ($e_i \to \theta_i$)"),
        mlines.Line2D([], [], color="black", lw=2.0, ls="--",
                      label=r"off-diag $R^2$ ($e_i \to \theta_j$, $i \neq j$)"),
        mlines.Line2D([], [], color="#FFF3CD", lw=8, alpha=0.6,
                      label="L4 phase transition"),
    ]
    leg1 = ax.legend(handles=family_handles, loc="lower right",
                     framealpha=0.9, fontsize=8, title="Family")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="upper left",
              framealpha=0.9, fontsize=7.5)

    ax.set_xlabel("Encoder layer")
    ax.set_ylabel(r"Ridge val $R^2$")
    ax.set_title(
        f"Cross-$\\theta$ probe by layer: per-dim subspace specialization "
        f"(n={len(by_task)} tasks; gap between solid \\& dashed = specialization)"
    )
    ax.set_xlim(-0.3, 11.5)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()
    out = FIGURES_DIR / "cross_theta_layer_trajectory.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}  ({len(by_task)} tasks)")

    # Print per-group L11 numbers for the caption.
    print("\n=== Group L11 means ===")
    for g in ("distr", "standard", "highdim"):
        if g not in group_stats:
            continue
        s = group_stats[g]
        print(f"  {g:<10}  diag={s['diag_mean'][-1]:.3f}  "
              f"off={s['off_mean'][-1]:.3f}  "
              f"gap={s['diag_mean'][-1] - s['off_mean'][-1]:+.3f}  "
              f"n={s['n_tasks']}")


if __name__ == "__main__":
    main()
