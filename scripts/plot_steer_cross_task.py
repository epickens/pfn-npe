"""Cross-task figure for CAA-style distractor steering.

Loads per-task npz output from steer_distractor_direction.py and produces:
  A. ||d[k]||  vs layer, all tasks on same axes.
  B. ΔR²     vs layer (ablation effect on mean probe), all tasks.
  C. ΔNLL    vs layer (ablation effect on variance probe), all tasks.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

STEER_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/steer")
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

TASKS = [
    ("two_moons_distractors", r"two_moons ($d_\theta$=2)"),
    ("sir_distractors", r"sir ($d_\theta$=2)"),
    ("bernoulli_glm_distractors", r"bernoulli_glm ($d_\theta$=10)"),
    ("gaussian_mixture_distractors", r"gaussian_mixture ($d_\theta$=2)"),
]
COLORS = ["#2E86AB", "#2CA02C", "#6B4C9A", "#E8573A"]
SEED = 42


def main() -> None:
    data = []
    for name, label in TASKS:
        p = STEER_DIR / f"{name}_s{SEED}.npz"
        if not p.exists():
            print(f"Missing: {p}")
            continue
        d = dict(np.load(p, allow_pickle=True))
        data.append((name, label, d))
    if not data:
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    layers = np.arange(12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: ΔR² vs layer (the main causal result)
    ax = axes[0]
    for (name, label, d), color in zip(data, COLORS, strict=False):
        ax.plot(layers, d["delta_r2"], "o-", color=color, ms=4, lw=1.5, label=label)
    ax.axhline(0, color="grey", ls=":", lw=0.7)
    ax.axvspan(1.5, 2.5, color="#D4EDDA", alpha=0.3, zorder=0)
    ax.set_xlabel("Ablation layer k")
    ax.set_ylabel(r"$\Delta$ R² (ablated − baseline)")
    ax.set_title("A. Mean probe effect of projecting out d[k]")
    ax.set_xlim(-0.3, 11.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7.5)

    # Panel B: ΔNLL vs layer
    ax = axes[1]
    for (name, label, d), color in zip(data, COLORS, strict=False):
        ax.plot(layers, d["delta_nll"], "o-", color=color, ms=4, lw=1.5, label=label)
    ax.axhline(0, color="grey", ls=":", lw=0.7)
    ax.axvspan(1.5, 2.5, color="#D4EDDA", alpha=0.3, zorder=0)
    ax.set_xlabel("Ablation layer k")
    ax.set_ylabel(r"$\Delta$ NLL (ablated − baseline)")
    ax.set_title("B. Variance probe effect of projecting out d[k]")
    ax.set_xlim(-0.3, 11.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7.5)

    # Panel C: ||d|| per layer parsed from log files
    ax = axes[2]
    import re
    for (name, label, d), color in zip(data, COLORS, strict=False):
        log = Path(f"logs/steer_{name}_s{SEED}.log")
        if not log.exists():
            continue
        norms = np.zeros(12)
        for line in log.read_text().splitlines():
            m = re.match(r"\s*layer\s+(\d+)\s+\|\|d\|\|=([0-9.]+)", line)
            if m:
                norms[int(m.group(1))] = float(m.group(2))
        ax.plot(layers, norms, "o-", color=color, ms=4, lw=1.5, label=label)
    ax.axvspan(1.5, 2.5, color="#D4EDDA", alpha=0.3, zorder=0)
    ax.set_xlabel("Layer k")
    ax.set_ylabel(r"$\|d[k]\|$ (fp32)")
    ax.set_title("C. Distractor direction norm per layer")
    ax.set_xlim(-0.3, 11.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7.5)

    fig.suptitle("CAA-style distractor ablation across tasks, seed 42")
    fig.tight_layout()
    out = FIG_DIR / "steer_cross_task.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
