"""Summary figures for the quantile-probe validation experiment.

Aggregates quantile_validate npz outputs across seeds and produces two
figures:

  quantile_validate_summary.png — cross-task 3 panels at best layer:
    A. Bar chart of val vs ref pinball, mean ± seed-range per task.
    B. Per-τ Pearson correlation, mean across seeds with shaded min/max
       band, one line per task.
    C. Per-τ RMSE (θ-space, linear), same structure as B.

  quantile_validate_per_layer.png — 2×3 grid, per-task correlation vs
  layer for each τ, with mean curve and per-seed traces. Shows where
  information emerges through the encoder depth.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VALIDATE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/quantile_validate")
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
    ("slcp_distractors", "slcp+distr"),
    ("two_moons_distractors", "two_moons+distr"),
    ("gaussian_mixture_distractors", "gaussian_mixture+distr"),
    ("bernoulli_glm_distractors", "bernoulli_glm+distr"),
    ("sir_distractors", "sir+distr"),
]
N_LAYERS = 12


def load_task_seeds(task: str) -> list[dict]:
    return [
        dict(np.load(p, allow_pickle=True))
        for p in sorted(VALIDATE_DIR.glob(f"{task}_s*.npz"))
    ]


def aggregate(task_runs: list[dict]) -> dict:
    best = [int(r["best_layer"]) for r in task_runs]
    return {
        "n_seeds": len(task_runs),
        "seeds": [int(r["seed"]) for r in task_runs],
        "best_layers": np.asarray(best),
        "val_pinball_best": np.asarray(
            [float(r["pinball_val"][b]) for r, b in zip(task_runs, best, strict=True)]
        ),
        "ref_pinball_best": np.asarray(
            [float(r["pinball_ref"][b]) for r, b in zip(task_runs, best, strict=True)]
        ),
        "corr_best": np.stack(
            [r["corr"][b] for r, b in zip(task_runs, best, strict=True)]
        ),
        "rmse_best": np.stack(
            [r["rmse"][b] for r, b in zip(task_runs, best, strict=True)]
        ),
        "corr_full": np.stack([r["corr"] for r in task_runs]),   # (n_seeds, N_LAYERS, n_tau)
        "pinball_val_full": np.stack([r["pinball_val"] for r in task_runs]),
        "pinball_ref_full": np.stack([r["pinball_ref"] for r in task_runs]),
        "taus": task_runs[0]["taus"],
    }


def plot_summary(per_task: dict[str, dict], labels: dict[str, str]) -> None:
    tasks = list(per_task.keys())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── A: pinball val/ref with per-seed error bars ──
    ax = axes[0]
    xs = np.arange(len(tasks))
    w = 0.38
    val_means = [per_task[t]["val_pinball_best"].mean() for t in tasks]
    val_std = [per_task[t]["val_pinball_best"].std() for t in tasks]
    ref_means = [per_task[t]["ref_pinball_best"].mean() for t in tasks]
    ref_std = [per_task[t]["ref_pinball_best"].std() for t in tasks]
    ax.bar(xs - w/2, val_means, w, yerr=val_std, color="C0",
           label="val pinball", capsize=3)
    ax.bar(xs + w/2, ref_means, w, yerr=ref_std, color="C3",
           label="ref pinball (10 obs)", capsize=3)
    # Per-seed scatter overlay
    for i, t in enumerate(tasks):
        for v in per_task[t]["val_pinball_best"]:
            ax.scatter(i - w/2, v, s=10, color="black", alpha=0.6, zorder=3)
        for v in per_task[t]["ref_pinball_best"]:
            ax.scatter(i + w/2, v, s=10, color="black", alpha=0.6, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels([labels[t] for t in tasks],
                                          rotation=20, ha="right")
    ax.set_ylabel("Pinball loss at best layer")
    n_total = sum(per_task[t]["n_seeds"] for t in tasks)
    ax.set_title(f"A. Generalization at best layer (n={n_total})")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # ── B: per-τ correlation with shaded band ──
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    for color, t in zip(colors, tasks, strict=False):
        corr = per_task[t]["corr_best"]                 # (n_seeds, n_tau)
        taus = per_task[t]["taus"]
        mean = corr.mean(axis=0)
        lo = corr.min(axis=0)
        hi = corr.max(axis=0)
        ax.plot(taus, mean, "o-", color=color, lw=1.8, label=labels[t])
        ax.fill_between(taus, lo, hi, color=color, alpha=0.18)
    ax.axhline(0, color="grey", ls=":", alpha=0.5)
    ax.set_xlabel("Quantile τ")
    ax.set_ylabel("Pearson r (predicted vs empirical)")
    ax.set_title("B. Per-τ correlation (mean ± seed range)")
    ax.legend(fontsize=7, loc="lower center", ncol=2)
    ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.05)

    # ── C: per-τ RMSE with shaded band ──
    ax = axes[2]
    for color, t in zip(colors, tasks, strict=False):
        rmse = per_task[t]["rmse_best"]
        taus = per_task[t]["taus"]
        mean = rmse.mean(axis=0)
        lo = rmse.min(axis=0)
        hi = rmse.max(axis=0)
        ax.plot(taus, mean, "s-", color=color, lw=1.8, label=labels[t])
        ax.fill_between(taus, lo, hi, color=color, alpha=0.18)
    ax.set_xlabel("Quantile τ"); ax.set_ylabel("RMSE (θ-space, linear)")
    ax.set_title("C. Per-τ RMSE (mean ± seed range)")
    ax.legend(fontsize=7, loc="upper center", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Quantile-probe validation across {len(tasks)} tasks × {n_total} total runs",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "quantile_validate_summary.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")


def plot_per_layer(per_task: dict[str, dict], labels: dict[str, str]) -> None:
    tasks = list(per_task.keys())
    ncols = 3
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows),
                             sharey=True)
    axes = np.atleast_1d(axes).flatten()
    layers = np.arange(N_LAYERS)

    for ax, t in zip(axes, tasks, strict=False):
        corr = per_task[t]["corr_full"]                 # (n_seeds, N_LAYERS, n_tau)
        taus = per_task[t]["taus"]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(taus)))
        for t_i, tau in enumerate(taus):
            mean = corr[:, :, t_i].mean(axis=0)
            lo = corr[:, :, t_i].min(axis=0)
            hi = corr[:, :, t_i].max(axis=0)
            ax.plot(layers, mean, "o-", color=cmap[t_i], lw=1.6,
                    label=f"τ={tau:.2f}", ms=4)
            ax.fill_between(layers, lo, hi, color=cmap[t_i], alpha=0.18)
        ax.axhline(0, color="grey", ls=":", alpha=0.5)
        best_mean = per_task[t]["best_layers"].mean()
        ax.axvline(best_mean, color="black", ls=":", alpha=0.5)
        ax.set_xlabel("Encoder layer"); ax.set_ylabel("Pearson r")
        ax.set_title(
            f"{labels[t]} (n={per_task[t]['n_seeds']} seeds)"
        )
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes[len(tasks):]:
        ax.set_axis_off()
    fig.suptitle("Per-τ correlation vs layer — validation on sbibm ref obs",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "quantile_validate_per_layer.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")


def main() -> None:
    per_task: dict[str, dict] = {}
    labels: dict[str, str] = {}
    for task, lab in TASK_ORDER:
        runs = load_task_seeds(task)
        if not runs:
            continue
        per_task[task] = aggregate(runs)
        labels[task] = lab

    if not per_task:
        print("No quantile_validate results found.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_summary(per_task, labels)
    plot_per_layer(per_task, labels)

    print("\n=== Per-task aggregated summary (at each run's best layer) ===")
    print(f"{'task':<35} {'n':>2} {'best_avg':>8} {'val_pin':>17} "
          f"{'ref_pin':>17} {'r_median':>17} {'r_mean':>17}")
    for t in per_task:
        a = per_task[t]
        taus = a["taus"]
        i_med = int(np.argmin(np.abs(taus - 0.5)))
        r_med = a["corr_best"][:, i_med]
        r_mean = a["corr_best"].mean(axis=1)
        print(
            f"{t:<35} {a['n_seeds']:>2} "
            f"{a['best_layers'].mean():>8.2f} "
            f"{a['val_pinball_best'].mean():>8.4f}±{a['val_pinball_best'].std():.4f} "
            f"{a['ref_pinball_best'].mean():>8.4f}±{a['ref_pinball_best'].std():.4f} "
            f"{r_med.mean():>+8.3f}±{r_med.std():.3f} "
            f"{r_mean.mean():>+8.3f}±{r_mean.std():.3f}"
        )


if __name__ == "__main__":
    main()
