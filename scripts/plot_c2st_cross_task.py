"""Cross-task comparison of per-layer C2ST.

Mirror of plot_probe_cross_task.py but with C2ST as the y-axis (lower
= better). Loads aggregated C2ST per task from
pfn_testing/sbi/outputs/layer_ablation/sweep/agg_{task}_s{seed}.npz
(except two_moons which keeps its existing legacy filename).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("pfn_testing/sbi/outputs")
ABL_DIR = OUT_DIR / "layer_ablation"
SWEEP_DIR = ABL_DIR / "sweep"
FIG_DIR = ABL_DIR / "figures"

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

# (task, label, npz path relative to SWEEP_DIR)
# Single seed (42) for fair cross-task comparison; two_moons uses its
# existing seed-42-only aggregate at the legacy filename.
TASKS = [
    ("two_moons_distractors",
     r"two_moons ($d_\theta$=2, 92 distractors)",
     "agg_two_moons_distractors_s42-123-7.npz"),
    ("gaussian_mixture_distractors",
     r"gaussian_mixture ($d_\theta$=2, 92 distractors)",
     "agg_gaussian_mixture_distractors_s42-123-7.npz"),
    ("bernoulli_glm_distractors",
     r"bernoulli_glm ($d_\theta$=10, 90 distractors)",
     "agg_bernoulli_glm_distractors_s42-123-7.npz"),
    ("sir_distractors",
     r"sir ($d_\theta$=2, 90 distractors)",
     "agg_sir_distractors_s42-123-7.npz"),
]
COLORS = ["#2E86AB", "#E8573A", "#6B4C9A", "#2CA02C"]


def load_agg(fname: str) -> dict | None:
    p = SWEEP_DIR / fname
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def main() -> None:
    data = [(name, label, load_agg(fname)) for name, label, fname in TASKS]
    present = [(n, l, d) for n, l, d in data if d is not None]
    missing = [n for n, _, d in data if d is None]
    if missing:
        print(f"Note: missing C2ST aggregates for {missing} — skipping those.")

    fig = plt.figure(figsize=(11.5, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.25)

    # ── Left: absolute C2ST per task ──
    ax1 = fig.add_subplot(gs[0, 0])
    layers = np.arange(12)
    for (name, label, d), color in zip(present, COLORS, strict=False):
        ax1.errorbar(layers, d["means"], yerr=d["stds"], fmt="o-",
                     color=color, ms=4, lw=1.5, capsize=3, label=label)
    ax1.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.7)
    ax1.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.4, zorder=0)
    ax1.set_xlabel("Encoder layer")
    ax1.set_ylabel("C2ST (lower = better)")
    ax1.set_title("A. C2ST by task")
    ax1.set_xlim(-0.3, 11.5)
    ax1.set_ylim(0.45, 1.02)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=7.5)

    # ── Right: gain over layer 0 (reduction in C2ST) ──
    # Plotted as positive numbers so taller = more improvement.
    ax2 = fig.add_subplot(gs[0, 1])
    for (name, label, d), color in zip(present, COLORS, strict=False):
        gain = d["means"][0] - d["means"]
        ax2.plot(layers, gain, "o-", color=color, ms=4, lw=1.5, label=label)
    ax2.axhline(0, color="grey", ls=":", lw=0.6, alpha=0.7)
    ax2.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.4, zorder=0)
    ax2.set_xlabel("Encoder layer")
    ax2.set_ylabel(r"$\Delta$ C2ST vs. layer 0 (reduction)")
    ax2.set_title("B. C2ST improvement over layer 0")
    ax2.set_xlim(-0.3, 11.5)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=7.5)

    out = FIG_DIR / "c2st_cross_task.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
