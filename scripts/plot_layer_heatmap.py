"""Heatmap of per-seed, per-layer C2ST values + per-observation strip plots."""
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
})

SEEDS = [42, 123, 7]
N_LAYERS = 12


def load_all() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-seed per-layer C2ST arrays.

    Returns:
        means: (n_seeds, 12)
        stds:  (n_seeds, 12)
        obs:   list of (n_seeds, 12, n_obs) — all observations
    """
    base = OUT_DIR / "two_moons_distractors"
    means = np.zeros((len(SEEDS), N_LAYERS))
    stds = np.zeros_like(means)
    all_obs: list[list[np.ndarray]] = [[] for _ in SEEDS]

    for si, seed in enumerate(SEEDS):
        for k in range(N_LAYERS):
            r = np.load(
                base / f"n10000_per_dim_regressor_layer{k}_s{seed}/results/results.npz",
                allow_pickle=True,
            )
            c = np.asarray(r["c2st_tabpfn"])
            means[si, k] = c.mean()
            stds[si, k] = c.std()
            all_obs[si].append(c)

    return means, stds, all_obs


def main() -> None:
    means, stds, all_obs = load_all()

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [1.2, 2]},
    )

    # ── Top: annotated heatmap ──
    ax = axes[0]
    im = ax.imshow(means, aspect="auto", cmap="RdYlGn_r", vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(SEEDS)))
    ax.set_yticklabels([f"seed {s}" for s in SEEDS])
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels(range(N_LAYERS))
    ax.set_xlabel("Encoder layer")
    ax.set_title("Per-seed C2ST (lower = better)")

    for si in range(len(SEEDS)):
        for k in range(N_LAYERS):
            val = means[si, k]
            color = "white" if val > 0.75 else "black"
            ax.text(k, si, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("C2ST", fontsize=8)

    # ── Bottom: strip plot of individual observations ──
    ax2 = axes[1]
    rng = np.random.default_rng(0)
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for si, seed in enumerate(SEEDS):
        for k in range(N_LAYERS):
            obs = all_obs[si][k]
            jitter = rng.uniform(-0.08, 0.08, size=len(obs)) + si * 0.25 - 0.25
            ax2.scatter(
                k + jitter, obs,
                s=12, alpha=0.5, color=colors[si],
                edgecolors="none",
                label=f"seed {seed}" if k == 0 else None,
            )

    agg = np.load(SWEEP_DIR / "agg_s42-123-7.npz")
    ax2.errorbar(
        np.arange(N_LAYERS), agg["means"], yerr=agg["stds"],
        fmt="D-", color="k", ms=5, lw=1.5, capsize=3, zorder=10,
        label="aggregate",
    )
    ax2.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.7)

    raw_m = float(agg["raw_mean"])
    ax2.axhline(raw_m, color="#888", ls="--", lw=0.8)
    ax2.text(11.4, raw_m, "raw\nbaseline", va="center", ha="left",
             fontsize=7, color="#888")

    # shade the transition zone
    ax2.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.5, zorder=0)
    ax2.text(4, 1.0, "transition", ha="center", va="top", fontsize=7,
             color="#B8860B", style="italic")

    ax2.set_xlabel("Encoder layer")
    ax2.set_ylabel("C2ST (lower = better)")
    ax2.set_title("Per-observation C2ST across seeds")
    ax2.set_xlim(-0.5, 12)
    ax2.set_ylim(0.42, 1.05)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax2.legend(loc="upper right", framealpha=0.9, ncol=4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(h_pad=2)
    out = FIG_DIR / "heatmap.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
