"""Summary figure for the quantile-probe experiment across tasks and seeds.

Loads quantile npz outputs for all (task, seed) combinations that exist and
produces a 4-panel figure:
  A. Per-task box plot of best-layer Δ pinball (probe − baseline). More
     negative = bigger improvement over the constant-quantile baseline.
  B. Per-task box plot of best-layer calibration error
     (mean |empirical_τ − τ| across τ).
  C. Per-task box plot of layer-range Δ pinball (max − min over layers) —
     a measure of layer localization.
  D. Best-layer IQR-CV (CV of predicted IQR across val examples) vs the
     CAA-steer max|ΔR²| (loaded from the steer outputs). Tests whether
     tasks where the probe surfaces large posterior-scale variation are
     also the ones sensitive to CAA steering.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

QUANT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/quantile")
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
    ("two_moons_distractors", "two_moons"),
    ("sir_distractors", "sir"),
    ("bernoulli_glm_distractors", "bernoulli_glm"),
    ("gaussian_mixture_distractors", "gaussian_mixture"),
    ("slcp_distractors", "slcp"),
]
SEEDS = [42, 123, 7]
SENSITIVITY_THRESHOLD = 0.1  # mirrors plot_steer_summary.py


def load_quant(task: str, seed: int) -> dict | None:
    p = QUANT_DIR / f"{task}_s{seed}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def load_steer(task: str, seed: int) -> dict | None:
    p = STEER_DIR / f"{task}_s{seed}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def main() -> None:
    rows = []
    for task, _ in TASKS:
        for seed in SEEDS:
            d = load_quant(task, seed)
            if d is None:
                continue
            best = int(d["best_layer"])
            taus = d["taus"]
            calib = d["calibration"][best]
            iqr = d["iqr_val"][best]                         # (n_val, dim_theta)
            iqr_mean = iqr.mean(axis=-1)                     # (n_val,) over θ-dim
            iqr_cv = (
                float(iqr_mean.std() / (abs(iqr_mean.mean()) + 1e-12))
                if iqr_mean.size > 0 else float("nan")
            )
            steer = load_steer(task, seed)
            steer_max = (
                float(np.max(np.abs(steer["delta_r2"]))) if steer is not None else None
            )
            rows.append({
                "task": task,
                "seed": seed,
                "best_layer": best,
                "delta_pinball_best": float(d["pinball"][best] - d["pinball_baseline"][best]),
                "pinball_range": float(d["pinball"].max() - d["pinball"].min()),
                "calib_err": float(np.mean(np.abs(calib - taus))),
                "iqr_cv": iqr_cv,
                "steer_max_abs_dr2": steer_max,
                "crossing_rate_best": float(d["crossing_rate"][best]),
            })

    if not rows:
        print("No quantile results found.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    task_names = [t for t, _ in TASKS if t in by_task]
    task_labels = {t: lab for t, lab in TASKS}

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    # ── A: Δ pinball best-layer (probe − baseline) ──
    ax = fig.add_subplot(gs[0, 0])
    vals = [[r["delta_pinball_best"] for r in by_task[t]] for t in task_names]
    bp = ax.boxplot(vals, tick_labels=[task_labels[t] for t in task_names],
                    patch_artist=True, widths=0.55)
    for patch, t in zip(bp["boxes"], task_names, strict=False):
        mean_eff = np.mean([abs(r["delta_pinball_best"]) for r in by_task[t]])
        patch.set_facecolor("#A3D9A3" if mean_eff > 0.02 else "#D9D9D9")
        patch.set_alpha(0.9)
    for seed in SEEDS:
        for i, t in enumerate(task_names, start=1):
            row = next((r for r in by_task[t] if r["seed"] == seed), None)
            if row:
                ax.scatter(i, row["delta_pinball_best"], s=14, color="black",
                           alpha=0.7, zorder=3)
    ax.axhline(0, color="grey", ls=":", alpha=0.7)
    ax.set_ylabel(r"$\Delta$ pinball (probe − baseline) at best layer")
    ax.set_title(f"A. Probe vs baseline at best layer (n={len(rows)})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # ── B: calibration error at best layer ──
    ax = fig.add_subplot(gs[0, 1])
    vals = [[r["calib_err"] for r in by_task[t]] for t in task_names]
    bp = ax.boxplot(vals, tick_labels=[task_labels[t] for t in task_names],
                    patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#A3C5F5")
        patch.set_alpha(0.9)
    for seed in SEEDS:
        for i, t in enumerate(task_names, start=1):
            row = next((r for r in by_task[t] if r["seed"] == seed), None)
            if row:
                ax.scatter(i, row["calib_err"], s=14, color="black",
                           alpha=0.7, zorder=3)
    ax.set_ylabel(r"mean $|$empirical $\tau$ − $\tau|$")
    ax.set_title("B. Calibration error at best layer")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # ── C: pinball range across layers ──
    ax = fig.add_subplot(gs[1, 0])
    vals = [[r["pinball_range"] for r in by_task[t]] for t in task_names]
    bp = ax.boxplot(vals, tick_labels=[task_labels[t] for t in task_names],
                    patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#F5C6A3")
        patch.set_alpha(0.9)
    for seed in SEEDS:
        for i, t in enumerate(task_names, start=1):
            row = next((r for r in by_task[t] if r["seed"] == seed), None)
            if row:
                ax.scatter(i, row["pinball_range"], s=14, color="black",
                           alpha=0.7, zorder=3)
    ax.set_ylabel(r"$\max_k$ pinball $- \min_k$ pinball")
    ax.set_title("C. Layer-localized improvement")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # ── D: IQR-CV vs steer max|ΔR²| ──
    ax = fig.add_subplot(gs[1, 1])
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_names)))
    plotted_any = False
    for color, t in zip(colors, task_names, strict=False):
        xs, ys = [], []
        for r in by_task[t]:
            if r["steer_max_abs_dr2"] is None:
                continue
            xs.append(r["iqr_cv"])
            ys.append(r["steer_max_abs_dr2"])
        if not xs:
            continue
        ax.scatter(xs, ys, color=color, s=70, edgecolor="black", lw=0.5,
                   label=task_labels[t], alpha=0.85)
        plotted_any = True
    if plotted_any:
        ax.axhline(SENSITIVITY_THRESHOLD, color="grey", ls=":", lw=0.7,
                   label="steer sensitivity threshold")
        ax.set_xlabel("Best-layer IQR-CV (probe-predicted IQR across val)")
        ax.set_ylabel(r"steer $\max_k \, |\Delta R^2|$")
        ax.set_title("D. Posterior-scale variation vs CAA-steer effect")
        ax.legend(loc="upper left", fontsize=7.5)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_axis_off()
        ax.set_title("D. Steer outputs missing — skipped")

    fig.suptitle(
        f"Quantile probe across {len(task_names)} tasks × {len(SEEDS)} seeds",
        fontsize=11,
    )
    out = FIG_DIR / "quantile_summary.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")

    print("\n=== Per (task, seed) summary ===")
    print(f"{'task':<35} {'seed':>4} {'best':>5} {'Δpinball':>10} "
          f"{'calib_err':>10} {'IQR_CV':>8} {'cross':>7}")
    for r in rows:
        print(f"{r['task']:<35} {r['seed']:>4} {r['best_layer']:>5} "
              f"{r['delta_pinball_best']:>+10.4f} {r['calib_err']:>10.4f} "
              f"{r['iqr_cv']:>8.3f} {r['crossing_rate_best']:>7.4f}")


if __name__ == "__main__":
    main()
