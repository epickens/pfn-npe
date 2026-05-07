"""Compact main-text marginal-vs-joint C2ST gap figure.

This is a space-saving companion to the full marginal/joint lollipop plot.
It shows only the quantity needed for the main diagnostic claim:
joint C2ST minus marginal C2ST for PFN-NPE at 10k simulations.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_IN = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_marginal_joint_gap.csv")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")

TASK_LABELS_SHORT = {
    "ar1_ts_t50": "AR(1)",
    "bernoulli_glm": "Bernoulli GLM",
    "bernoulli_glm_distractors": "Bernoulli + distr.",
    "gaussian_linear": "Gaussian lin.",
    "gaussian_linear_uniform": "Gaussian lin. unif.",
    "gaussian_mixture": "Gaussian mix.",
    "gaussian_mixture_distractors": "Gaussian mix. + distr.",
    "lotka_volterra": "Lotka-Volterra",
    "ou": "OU",
    "sir": "SIR",
    "sir_distractors": "SIR + distr.",
    "slcp": "SLCP",
    "slcp_distractors": "SLCP + distr.",
    "solar_dynamo": "Solar dynamo",
    "two_moons": "Two moons",
    "two_moons_distractors": "Two moons + distr.",
}

DISTRACTOR_TASKS = {
    "two_moons_distractors",
    "gaussian_mixture_distractors",
    "sir_distractors",
    "slcp_distractors",
    "bernoulli_glm_distractors",
}
HIGHDIM_TASKS = {"ou", "solar_dynamo", "ar1_ts_t50", "lotka_volterra"}
GROUP_COLORS = {"standard": "#009E73", "distr": "#0072B2", "highdim": "#D55E00"}
GROUP_LABELS = {
    "standard": "standard SBIBM",
    "distr": "distractor variants",
    "highdim": "high-dim/time-series",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def task_group(task: str) -> str:
    if task in DISTRACTOR_TASKS:
        return "distr"
    if task in HIGHDIM_TASKS:
        return "highdim"
    return "standard"


def main() -> None:
    rows = pd.read_csv(CSV_IN)
    rows = rows.sort_values("gap_mean", ascending=True).reset_index(drop=True)
    labels = [TASK_LABELS_SHORT.get(task, task.replace("_", " ")) for task in rows["task"]]
    gaps = rows["gap_mean"].to_numpy(dtype=float)
    gap_sd = rows["gap_sd"].to_numpy(dtype=float)
    groups = [task_group(task) for task in rows["task"]]
    colors = [GROUP_COLORS[group] for group in groups]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.0)
    ax.barh(y, gaps, xerr=gap_sd, color=colors, alpha=0.9, height=0.62, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Joint C2ST - marginal C2ST")
    ax.set_xlim(0.0, max(0.25, float(np.nanmax(gaps + gap_sd)) + 0.015))
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(axis="y", length=0)

    median_gap = float(np.median(gaps))
    mean_gap = float(np.mean(gaps))
    ax.text(
        0.99,
        0.04,
        f"all {len(rows)} gaps > 0; median={median_gap:.2f}, mean={mean_gap:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.25",
    )

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=6, color=GROUP_COLORS[group])
        for group in ("standard", "distr", "highdim")
    ]
    ax.legend(
        handles,
        [GROUP_LABELS[group] for group in ("standard", "distr", "highdim")],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.12),
        frameon=True,
        framealpha=0.92,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = FIG_DIR / f"c2st_gap_compact.{suffix}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
