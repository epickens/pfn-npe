"""Summary figure for CAA steering experiment across tasks and seeds.

Loads steer npz outputs for all (task, seed) combinations that exist and
produces a 4-panel figure:
  A. Per-task box plot of max|ΔR²| across seeds.
  B. Per-task box plot of (max − min) ΔR² across seeds.
  C. Scatter of max|ΔR²| vs baseline R² per (task, seed).
  D. Scatter of max|ΔR²| vs posterior-variance CV (task-level).

The goal is to surface whether the ablation effect splits into groups
(sensitive vs quiet tasks) and whether baseline R² or posterior-variance
structure predicts that split.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
    ("two_moons_distractors", r"two_moons"),
    ("sir_distractors", r"sir"),
    ("bernoulli_glm_distractors", r"bernoulli_glm"),
    ("gaussian_mixture_distractors", r"gaussian_mixture"),
    ("slcp_distractors", r"slcp"),
]
SEEDS = [42, 123, 7]

# Posterior-variance CV from analyze_posterior_variance_cv.py (hard-coded;
# computed from sbibm reference posterior samples)
CV_POST_VAR = {
    "two_moons_distractors": 0.98,
    "sir_distractors": 1.22,
    "bernoulli_glm_distractors": 0.45,
    "gaussian_mixture_distractors": 0.18,
    "slcp_distractors": 0.72,
}

# Empirical grouping from the steering data itself: the max|ΔR²| distribution
# separates cleanly into two clusters with a ~5–10× gap.
SENSITIVITY_THRESHOLD = 0.1


def load_steer(task: str, seed: int) -> dict | None:
    p = STEER_DIR / f"{task}_s{seed}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def main() -> None:
    rows = []
    for task, _ in TASKS:
        for seed in SEEDS:
            d = load_steer(task, seed)
            if d is None:
                continue
            dr2 = d["delta_r2"]
            rows.append({
                "task": task,
                "seed": seed,
                "max_abs_dr2": float(np.max(np.abs(dr2))),
                "max_minus_min_dr2": float(np.max(dr2) - np.min(dr2)),
                "baseline_r2": float(d["baseline_r2"]),
                "baseline_nll": float(d["baseline_nll"]),
                "l0_dr2": float(dr2[0]),
                "l11_dr2": float(dr2[-1]),
            })

    if not rows:
        print("No steer results found.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Group by task
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    task_names = [t for t, _ in TASKS if t in by_task]
    task_labels = {t: lab for t, lab in TASKS}

    # ── A: box plot of max|ΔR²| per task ──
    ax = fig.add_subplot(gs[0, 0])
    vals = [[r["max_abs_dr2"] for r in by_task[t]] for t in task_names]
    bp = ax.boxplot(vals, tick_labels=[task_labels[t] for t in task_names],
                    patch_artist=True, widths=0.55)
    for patch, t in zip(bp["boxes"], task_names, strict=False):
        # Color by empirical sensitivity: sensitive = red, quiet = blue.
        mean_effect = np.mean([r["max_abs_dr2"] for r in by_task[t]])
        patch.set_facecolor("#F5A3A3" if mean_effect > SENSITIVITY_THRESHOLD else "#A3C5F5")
        patch.set_alpha(0.9)
    for seed in SEEDS:
        for i, t in enumerate(task_names, start=1):
            row = next((r for r in by_task[t] if r["seed"] == seed), None)
            if row:
                ax.scatter(i, row["max_abs_dr2"], s=14, color="black", alpha=0.7, zorder=3)
    ax.set_ylabel(r"$\max_k \, |\Delta R^2|$")
    ax.set_title(f"A. Max ablation effect per task (n={len(rows)})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # ── B: box plot of (max − min) ΔR² per task ──
    ax = fig.add_subplot(gs[0, 1])
    vals = [[r["max_minus_min_dr2"] for r in by_task[t]] for t in task_names]
    bp = ax.boxplot(vals, tick_labels=[task_labels[t] for t in task_names],
                    patch_artist=True, widths=0.55)
    for patch, t in zip(bp["boxes"], task_names, strict=False):
        # Color by empirical sensitivity: sensitive = red, quiet = blue.
        mean_effect = np.mean([r["max_abs_dr2"] for r in by_task[t]])
        patch.set_facecolor("#F5A3A3" if mean_effect > SENSITIVITY_THRESHOLD else "#A3C5F5")
        patch.set_alpha(0.9)
    for seed in SEEDS:
        for i, t in enumerate(task_names, start=1):
            row = next((r for r in by_task[t] if r["seed"] == seed), None)
            if row:
                ax.scatter(i, row["max_minus_min_dr2"], s=14, color="black",
                           alpha=0.7, zorder=3)
    ax.set_ylabel(r"$\max_k \Delta R^2 - \min_k \Delta R^2$")
    ax.set_title("B. ΔR² range across layers")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # ── C: scatter vs baseline R² ──
    ax = fig.add_subplot(gs[1, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_names)))
    for color, t in zip(colors, task_names, strict=False):
        xs = [r["baseline_r2"] for r in by_task[t]]
        ys = [r["max_abs_dr2"] for r in by_task[t]]
        ax.scatter(xs, ys, color=color, s=70, edgecolor="black", lw=0.5,
                   label=task_labels[t], alpha=0.85)
    ax.axhline(SENSITIVITY_THRESHOLD, color="grey", ls=":", lw=0.7,
               label="sensitivity threshold")
    ax.set_xlabel("Baseline R² (distract, no ablation)")
    ax.set_ylabel(r"$\max_k \, |\Delta R^2|$")
    ax.set_title("C. Ablation effect vs baseline probe R²")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.5)

    # ── D: scatter vs posterior-variance CV ──
    ax = fig.add_subplot(gs[1, 1])
    for color, t in zip(colors, task_names, strict=False):
        cv = CV_POST_VAR.get(t)
        if cv is None:
            continue
        ys = [r["max_abs_dr2"] for r in by_task[t]]
        ax.scatter([cv] * len(ys), ys, color=color, s=70,
                   edgecolor="black", lw=0.5, label=task_labels[t], alpha=0.85)
    ax.set_xlabel("CV of posterior Var across ref obs")
    ax.set_ylabel(r"$\max_k \, |\Delta R^2|$")
    ax.set_title("D. Ablation effect vs posterior-variance CV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.5)

    fig.suptitle(f"Distractor steering across {len(task_names)} tasks × {len(SEEDS)} seeds",
                 fontsize=11)
    out = FIG_DIR / "steer_summary.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")

    # Summary table
    print("\n=== Per (task, seed) summary ===")
    print(f"{'task':<35} {'seed':>4} {'max|ΔR²|':>10} {'range':>8} {'baseR²':>8}")
    for r in rows:
        print(f"{r['task']:<35} {r['seed']:>4} {r['max_abs_dr2']:>10.4f} "
              f"{r['max_minus_min_dr2']:>8.4f} {r['baseline_r2']:>8.4f}")


if __name__ == "__main__":
    main()
