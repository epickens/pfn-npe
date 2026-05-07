"""Cross-task comparison of per-layer Gaussian NLL from the variance probe.

Auto-loads every `{task}_s{seed}.npz` in the probe output directory and
aggregates across seeds (mean ± std). Tasks are auto-grouped by best-layer
ΔNLL into:
  complex (ΔNLL ≤ −0.1):     strong heteroscedastic signal
  gaussian_like (−0.1 < ΔNLL ≤ −0.01)
  anomaly (ΔNLL > −0.01):    near-zero heteroscedastic gain
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

GROUP_COLORS = {"complex": "#2E86AB", "gaussian_like": "#888888", "anomaly": "#E8573A"}
GROUP_LABELS = {
    "complex": r"complex posterior ($\Delta$NLL $\leq$ -0.1)",
    "gaussian_like": r"$\approx$Gaussian posterior (-0.1 $<$ $\Delta$NLL $\leq$ -0.01)",
    "anomaly": r"anomaly ($\Delta$NLL $>$ -0.01)",
}


def assign_group(best_delta_nll: float) -> str:
    if best_delta_nll <= -0.1:
        return "complex"
    if best_delta_nll <= -0.01:
        return "gaussian_like"
    return "anomaly"


def load_all() -> dict[str, dict]:
    by_task: dict[str, list[dict]] = {}
    for f in sorted(PROBE_DIR.glob("*.npz")):
        m = NAME_RE.match(f.name)
        if not m:
            continue
        task = m.group("task")
        d = np.load(str(f), allow_pickle=True)
        if "nll" not in d.files:
            continue
        by_task.setdefault(task, []).append({
            "nll": np.asarray(d["nll"]),
            "nll_homo": np.asarray(d["nll_homo"]),
            "nll_per_dim": np.asarray(d["nll_per_dim"]),
            "seed": int(d["seed"]),
        })
    out = {}
    for task, runs in by_task.items():
        nll_arr = np.stack([r["nll"] for r in runs])
        homo_arr = np.stack([r["nll_homo"] for r in runs])
        gap_arr = nll_arr - homo_arr
        per_dim = np.stack([r["nll_per_dim"] for r in runs])
        D = per_dim.shape[2]
        best_delta = float(gap_arr.mean(0).min())  # most negative mean across seeds
        out[task] = {
            "nll_mean": nll_arr.mean(0),
            "homo_mean": homo_arr.mean(0),
            "gap_mean": gap_arr.mean(0),
            "gap_std": gap_arr.std(0),
            "per_dim_mean": per_dim.mean(0),
            "n_seeds": len(runs),
            "D": D,
            "best_delta_nll": best_delta,
            "group": assign_group(best_delta),
            "label": f"{task} ($d_\\theta$={D})",
        }
    return out


def main() -> None:
    data = load_all()
    if not data:
        raise SystemExit(f"No variance-probe npzs in {PROBE_DIR}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    layers = np.arange(12)

    # Sort by group then ΔNLL (most negative first within group).
    group_order = {"complex": 0, "gaussian_like": 1, "anomaly": 2}
    tasks_sorted = sorted(
        data.keys(),
        key=lambda t: (group_order[data[t]["group"]], data[t]["best_delta_nll"]),
    )

    # Group-mean stats: mean and std across tasks within group.
    by_group: dict[str, list[np.ndarray]] = {}
    for task, d in data.items():
        by_group.setdefault(d["group"], []).append(d["gap_mean"])
    group_stats = {
        g: {"mean": np.stack(c).mean(0), "std": np.stack(c).std(0),
            "n_tasks": len(c)}
        for g, c in by_group.items()
    }

    # ── Cross-task summary: group means in foreground, per-task in background ──
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for task, d in sorted(data.items()):
        ax.plot(layers, d["gap_mean"], color=GROUP_COLORS[d["group"]],
                lw=0.7, alpha=0.25)

    for g in ("complex", "gaussian_like", "anomaly"):
        if g not in group_stats:
            continue
        s = group_stats[g]
        color = GROUP_COLORS[g]
        ax.plot(layers, s["mean"], color=color, lw=2.6, alpha=0.95,
                label=f"{GROUP_LABELS[g]} (n={s['n_tasks']} tasks)")
        ax.fill_between(layers, s["mean"] - s["std"], s["mean"] + s["std"],
                        color=color, alpha=0.18, linewidth=0)

    ax.axhline(0, color="grey", ls=":", lw=0.8)
    ax.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.4, zorder=0,
               label="L4 phase transition")
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel(r"$\Delta$ NLL vs. homoscedastic (negative = probe wins)")
    ax.set_title(
        f"Heteroscedastic-probe gain by task family "
        f"(n={len(data)} tasks; thick = group mean $\\pm$ task std, thin = per-task)"
    )
    ax.set_xlim(-0.3, 11.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "variance_probe_cross_task.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}  ({len(data)} tasks)")

    # ── Per-task detail grid ──
    n = len(tasks_sorted)
    if n == 0:
        return
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows),
                              squeeze=False)
    for i, task in enumerate(tasks_sorted):
        ax = axes[i // ncols, i % ncols]
        d = data[task]
        color = GROUP_COLORS[d["group"]]
        per_dim = d["per_dim_mean"]
        ax.plot(layers, d["nll_mean"], "o-", color="k", ms=3, lw=1.2,
                label=r"mean across $\theta$")
        ax.plot(layers, d["homo_mean"], "s--", color="grey", ms=2, lw=0.8,
                label="homoscedastic")
        a = max(0.25, 1.0 / per_dim.shape[1])
        for j in range(per_dim.shape[1]):
            ax.plot(layers, per_dim[:, j], ".-", color=color, alpha=a,
                    lw=0.6, ms=2)
        ax.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.4, zorder=0)
        ax.set_title(d["label"], fontsize=8)
        ax.set_xlim(-0.3, 11.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        if i % ncols == 0:
            ax.set_ylabel("Gaussian NLL")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Encoder layer")
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=6.5)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig2.tight_layout()
    out2 = FIG_DIR / "variance_probe_cross_task_per_dim.png"
    fig2.savefig(str(out2), bbox_inches="tight")
    print(f"Wrote {out2}")

    # ── Sanity print: groupings ──
    print("\n=== Group assignments ===")
    for task in tasks_sorted:
        d = data[task]
        print(f"  {task:<32} D={d['D']:>2}  best ΔNLL={d['best_delta_nll']:+.3f}  "
              f"-> {d['group']}")


if __name__ == "__main__":
    main()
