"""Publication-quality multi-panel summary of the layer ablation experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("pfn_testing/sbi/outputs")
ABL_DIR = OUT_DIR / "layer_ablation"
SWEEP_DIR = ABL_DIR / "sweep"
PROBE_DIR = ABL_DIR / "probe"
FIG_DIR = ABL_DIR / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LAYERS = np.arange(12)
SEED_COLORS = {"42": "#2E86AB", "123": "#A23B72", "7": "#F18F01"}
C0 = "#2E86AB"
C1 = "#E8573A"
C2 = "#F18F01"
C3 = "#6B4C9A"


def load_per_seed_c2st(seed: int) -> tuple[np.ndarray, np.ndarray]:
    base = OUT_DIR / "two_moons_distractors"
    means, stds = [], []
    for k in range(12):
        r = np.load(base / f"n10000_per_dim_regressor_layer{k}_s{seed}/results/results.npz",
                     allow_pickle=True)
        c = np.asarray(r["c2st_tabpfn"])
        means.append(c.mean()); stds.append(c.std())
    return np.array(means), np.array(stds)


def panel_a(ax: plt.Axes) -> None:
    """3-seed aggregate C2ST with per-seed traces."""
    agg = np.load(SWEEP_DIR / "agg_s42-123-7.npz")

    for seed, color in SEED_COLORS.items():
        m, s = load_per_seed_c2st(int(seed))
        ax.plot(LAYERS, m, ".-", color=color, alpha=0.35, lw=0.8, ms=3,
                label=f"seed {seed}")

    ax.errorbar(LAYERS, agg["means"], yerr=agg["stds"], fmt="o-", color="k",
                capsize=3, ms=4, lw=1.5, label="aggregate", zorder=5)
    ax.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.7)

    raw_m, raw_s = float(agg["raw_mean"]), float(agg["raw_std"])
    ax.axhspan(raw_m - raw_s, raw_m + raw_s, color="#D4D4D4", alpha=0.4, zorder=0)
    ax.axhline(raw_m, color="#888", ls="--", lw=0.8, zorder=0)
    ax.text(11.3, raw_m, "raw", va="center", ha="left", fontsize=7, color="#888")

    ax.set_xlabel("Encoder layer")
    ax.set_ylabel("C2ST (lower = better)")
    ax.set_title("A. Layer sweep (3 seeds, target pool)")
    ax.set_xlim(-0.3, 11.7)
    ax.set_ylim(0.45, 1.01)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="lower left", framealpha=0.9, ncol=2)


def panel_b(ax: plt.Axes) -> None:
    """C2ST vs probe R² — twin axes."""
    agg = np.load(SWEEP_DIR / "agg_s42-123-7.npz")
    probe = np.load(PROBE_DIR / "two_moons_distractors_s42.npz")

    ax.errorbar(LAYERS, agg["means"], yerr=agg["stds"], fmt="o-", color=C0,
                capsize=3, ms=4, lw=1.3, label="C2ST (flow)")
    ax.set_ylabel("C2ST", color=C0)
    ax.tick_params(axis="y", labelcolor=C0)
    ax.set_ylim(0.45, 1.01)
    ax.axhline(0.5, color=C0, ls=":", lw=0.5, alpha=0.5)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(LAYERS, probe["r2"], "s-", color=C1, ms=4, lw=1.3,
             label="Probe R² (ridge)")
    ax2.set_ylabel("Ridge R²", color=C1)
    ax2.tick_params(axis="y", labelcolor=C1)
    ax2.set_ylim(-0.05, 0.55)
    ax2.axhline(0, color=C1, ls=":", lw=0.5, alpha=0.5)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="center left", framealpha=0.9)

    ax.set_xlabel("Encoder layer")
    ax.set_title("B. C2ST vs linear probe")
    ax.set_xlim(-0.3, 11.7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))


def panel_c(ax: plt.Axes) -> None:
    """Target vs mean pooling."""
    t = np.load(SWEEP_DIR / "agg_s42.npz")
    m = np.load(SWEEP_DIR / "agg_mean_s42.npz")

    ax.errorbar(LAYERS, t["means"], yerr=t["stds"], fmt="o-", color=C0,
                capsize=3, ms=4, lw=1.3, label="target pool")
    ax.errorbar(LAYERS + 0.15, m["means"], yerr=m["stds"], fmt="s-", color=C2,
                capsize=3, ms=4, lw=1.3, label="mean pool")
    ax.axhline(0.5, color="grey", ls=":", lw=0.6, alpha=0.7)

    raw_m = float(t["raw_mean"])
    ax.axhline(raw_m, color="#888", ls="--", lw=0.8)
    ax.text(11.3, raw_m, "raw", va="center", ha="left", fontsize=7, color="#888")

    ax.set_xlabel("Encoder layer")
    ax.set_ylabel("C2ST (lower = better)")
    ax.set_title("C. Pooling comparison (seed 42)")
    ax.set_xlim(-0.3, 11.7)
    ax.set_ylim(0.45, 1.01)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="lower left", framealpha=0.9)


def panel_d(ax: plt.Axes) -> None:
    """Probe R² per dimension."""
    probe = np.load(PROBE_DIR / "two_moons_distractors_s42.npz")
    per_dim = probe["per_dim"]
    dim_labels = [r"$\theta_1$", r"$\theta_2$"]
    colors = [C3, "#2CA02C"]

    ax.plot(LAYERS, probe["r2"], "o-", color="k", ms=4, lw=1.5,
            label="mean R²", zorder=5)
    for d in range(per_dim.shape[1]):
        ax.plot(LAYERS, per_dim[:, d], ".-", color=colors[d], alpha=0.7,
                ms=5, lw=1, label=dim_labels[d])

    ax.axhline(0, color="grey", ls=":", lw=0.6, alpha=0.7)

    # annotate the transition
    ax.annotate("", xy=(4, 0.454), xytext=(3, 0.005),
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.2))
    ax.text(3.2, 0.22, "phase\ntransition", fontsize=7, color="#999",
            ha="center", style="italic")

    ax.set_xlabel("Encoder layer")
    ax.set_ylabel("Ridge val R²")
    ax.set_title(r"D. Linear probe per $\theta$ dimension")
    ax.set_xlim(-0.3, 11.7)
    ax.set_ylim(-0.05, 0.55)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="lower right", framealpha=0.9)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.tight_layout(h_pad=2.5, w_pad=2.5)

    out = FIG_DIR / "summary.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")

    out_pdf = FIG_DIR / "summary.pdf"
    fig.savefig(str(out_pdf), bbox_inches="tight")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
