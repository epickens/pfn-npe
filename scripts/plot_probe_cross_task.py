"""Cross-task comparison of per-layer linear-probe R².

Auto-loads every `{task}_s{seed}.npz` in the probe output directory,
aggregates across seeds (mean ± std), and produces:
  1. Single-panel cross-task summary, color-grouped by task family
     (distractor variants / standard SBIBM / high-dim time-series).
  2. Per-task detail grid with per-θ-dim traces.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("pfn_testing/sbi/outputs")
ABL_DIR = OUT_DIR / "layer_ablation"
PROBE_DIR = ABL_DIR / "probe"
FIG_DIR = ABL_DIR / "figures"

NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)\.npz$")

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


def load_all() -> dict[str, dict]:
    """Return {task: {"r2_mean": (L,), "r2_std": (L,), "per_dim_mean": (L,D),
    "n_seeds": int, "label": str}}."""
    by_task: dict[str, list[dict]] = {}
    for f in sorted(PROBE_DIR.glob("*.npz")):
        m = NAME_RE.match(f.name)
        if not m:
            continue
        task = m.group("task")
        d = np.load(str(f), allow_pickle=True)
        by_task.setdefault(task, []).append({
            "r2": np.asarray(d["r2"]),
            "per_dim": np.asarray(d["per_dim"]),
            "seed": int(d["seed"]),
        })
    out = {}
    for task, runs in by_task.items():
        r2_arr = np.stack([r["r2"] for r in runs])  # (n_seeds, L)
        # per_dim shapes vary by task (different D); stack and mean across seeds
        per_dim = np.stack([r["per_dim"] for r in runs])  # (n_seeds, L, D)
        D = per_dim.shape[2]
        label = f"{task} ($d_\\theta$={D})"
        out[task] = {
            "r2_mean": r2_arr.mean(0),
            "r2_std": r2_arr.std(0),
            "per_dim_mean": per_dim.mean(0),
            "n_seeds": len(runs),
            "D": D,
            "label": label,
        }
    return out


def main() -> None:
    data = load_all()
    if not data:
        raise SystemExit(f"No probe npzs in {PROBE_DIR}")

    # Drop gaussian_linear from the cross-task headline plot --- analytic
    # posterior is already linear at L0, so the curve is flat at high R²
    # and distracts from the L4 transition story. Still appears in the
    # per-dim detail grid below.
    data_main = {t: d for t, d in data.items() if t != "gaussian_linear"}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    layers = np.arange(12)

    # Group-mean stats: mean and std across tasks within group.
    by_group: dict[str, list[np.ndarray]] = {}
    for task, d in data_main.items():
        by_group.setdefault(task_group(task), []).append(d["r2_mean"])
    group_stats = {
        g: {"mean": np.stack(c).mean(0), "std": np.stack(c).std(0),
            "n_tasks": len(c)}
        for g, c in by_group.items()
    }

    # ── Cross-task summary: group means in foreground, per-task in background ──
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Per-task seed-mean lines, alpha'd into the background.
    for task, d in sorted(data_main.items()):
        ax.plot(layers, d["r2_mean"], color=GROUP_COLORS[task_group(task)],
                lw=0.7, alpha=0.25)

    # Group-mean lines + ±1 std band across tasks within group.
    for g in ("distr", "standard", "highdim"):
        if g not in group_stats:
            continue
        s = group_stats[g]
        color = GROUP_COLORS[g]
        ax.plot(layers, s["mean"], color=color, lw=2.6, alpha=0.95,
                label=f"{GROUP_LABELS[g]} (n={s['n_tasks']} tasks)")
        ax.fill_between(layers, s["mean"] - s["std"], s["mean"] + s["std"],
                        color=color, alpha=0.18, linewidth=0)

    ax.axhline(0, color="grey", ls=":", lw=0.6, alpha=0.7)
    ax.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.4, zorder=0,
               label="L4 phase transition")
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel(r"Ridge val $R^2$ (mean across $\theta$ dims)")
    ax.set_title(
        f"Linear probe $R^2$ by task family "
        f"(n={len(data_main)} tasks; thick = group mean $\\pm$ task std, "
        f"thin = per-task)"
    )
    ax.set_xlim(-0.3, 11.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "probe_cross_task.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}  ({len(data_main)} tasks plotted, gaussian_linear excluded)")

    # ── Per-task detail grid: one subplot per task, per-θ-dim traces ──
    tasks_sorted = sorted(data.keys(), key=lambda t: (task_group(t), t))
    n = len(tasks_sorted)
    if n == 0:
        return
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows),
                              squeeze=False)
    for i, task in enumerate(tasks_sorted):
        ax = axes[i // ncols, i % ncols]
        g = task_group(task)
        color = GROUP_COLORS[g]
        d = data[task]
        per_dim = d["per_dim_mean"]
        ax.plot(layers, d["r2_mean"], "o-", color="k", ms=3, lw=1.2,
                label=r"mean across $\theta$")
        for j in range(per_dim.shape[1]):
            alpha = max(0.25, 1.0 / per_dim.shape[1])
            ax.plot(layers, per_dim[:, j], ".-", color=color, alpha=alpha,
                    lw=0.6, ms=2)
        ax.axhline(0, color="grey", ls=":", lw=0.6, alpha=0.7)
        ax.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.4, zorder=0)
        ax.set_title(d["label"], fontsize=8)
        ax.set_xlim(-0.3, 11.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        if i % ncols == 0:
            ax.set_ylabel(r"Ridge val $R^2$")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Encoder layer")
        if i == 0:
            ax.legend(loc="lower right", framealpha=0.9, fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig2.tight_layout()
    out2 = FIG_DIR / "probe_cross_task_per_dim.png"
    fig2.savefig(str(out2), bbox_inches="tight")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
