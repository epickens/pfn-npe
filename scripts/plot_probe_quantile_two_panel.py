"""Two-panel layer-wise probe summary.

Panel A summarizes mean-parameter linear decoding from PFN embeddings.
Panel B summarizes marginal quantile decoding with pinball loss normalized by
the empirical-reference posterior quantiles for each task/seed.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ABL_DIR = Path("pfn_testing/sbi/outputs/layer_ablation")
PROBE_DIR = ABL_DIR / "probe"
VALIDATE_DIR = ABL_DIR / "quantile_validate"
FIG_DIR = ABL_DIR / "figures"
CSV_OUT = ABL_DIR / "probe_quantile_layer_summary.csv"

NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)\.npz$")
N_LAYERS = 12

TASK_ORDER = [
    "gaussian_linear",
    "gaussian_linear_uniform",
    "two_moons",
    "gaussian_mixture",
    "sir",
    "slcp",
    "bernoulli_glm",
    "two_moons_distractors",
    "gaussian_mixture_distractors",
    "sir_distractors",
    "slcp_distractors",
    "bernoulli_glm_distractors",
    "ou",
    "solar_dynamo",
    "ar1_ts_t50",
    "lotka_volterra",
]

DISTRACTOR_TASKS = {
    "two_moons_distractors",
    "gaussian_mixture_distractors",
    "sir_distractors",
    "slcp_distractors",
    "bernoulli_glm_distractors",
}
HIGHDIM_TASKS = {"ou", "solar_dynamo", "ar1_ts_t50", "lotka_volterra"}
GROUPS = ("standard", "distr", "highdim")
GROUP_COLORS = {"distr": "#0072B2", "standard": "#009E73", "highdim": "#D55E00"}
GROUP_LABELS = {
    "standard": "standard SBIBM",
    "distr": "distractor variants",
    "highdim": "high-dim/time-series",
}

plt.rcParams.update(
    {
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
    }
)


def task_group(task: str) -> str:
    if task in DISTRACTOR_TASKS:
        return "distr"
    if task in HIGHDIM_TASKS:
        return "highdim"
    return "standard"


def pinball_loss(theta_true: np.ndarray, quantiles: np.ndarray, taus: np.ndarray) -> float:
    """Mean pinball loss for quantiles with shape (n_ref, n_tau, dim_theta)."""
    err = theta_true[:, None, :] - quantiles
    loss = np.maximum(taus[None, :, None] * err, (taus[None, :, None] - 1.0) * err)
    return float(loss.mean())


def load_mean_probe() -> dict[str, np.ndarray]:
    runs_by_task: dict[str, list[np.ndarray]] = {}
    task_set = set(TASK_ORDER)
    for path in sorted(PROBE_DIR.glob("*_s*.npz")):
        match = NAME_RE.match(path.name)
        if match is None:
            continue
        task = match.group("task")
        if task not in task_set:
            continue
        loaded = np.load(path, allow_pickle=True)
        if "r2" not in loaded:
            continue
        r2 = np.asarray(loaded["r2"], dtype=float)
        if r2.shape != (N_LAYERS,):
            raise ValueError(f"Expected {N_LAYERS} r2 values in {path}, got {r2.shape}")
        runs_by_task.setdefault(task, []).append(r2)

    return {task: np.stack(runs).mean(axis=0) for task, runs in runs_by_task.items()}


def load_quantile_pinball_ratio() -> dict[str, np.ndarray]:
    runs_by_task: dict[str, list[np.ndarray]] = {}
    task_set = set(TASK_ORDER)
    for path in sorted(VALIDATE_DIR.glob("*_s*.npz")):
        loaded = np.load(path, allow_pickle=True)
        task = str(loaded["task"])
        if task not in task_set:
            continue
        ref_pinball = pinball_loss(
            theta_true=np.asarray(loaded["theta_true"], dtype=float),
            quantiles=np.asarray(loaded["emp_q"], dtype=float),
            taus=np.asarray(loaded["taus"], dtype=float),
        )
        if ref_pinball <= 1e-12:
            continue
        ratio = np.asarray(loaded["pinball_ref"], dtype=float) / ref_pinball
        if ratio.shape != (N_LAYERS,):
            raise ValueError(f"Expected {N_LAYERS} pinball values in {path}, got {ratio.shape}")
        runs_by_task.setdefault(task, []).append(ratio)

    return {task: np.stack(runs).mean(axis=0) for task, runs in runs_by_task.items()}


def group_curves(data: dict[str, np.ndarray], tasks: list[str]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {group: [] for group in GROUPS}
    for task in tasks:
        grouped[task_group(task)].append(data[task])
    return {group: np.stack(curves) for group, curves in grouped.items() if curves}


def plot_grouped_curves(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    tasks: list[str],
    *,
    ylabel: str,
    title: str,
    reference_y: float | None = None,
    log_y: bool = False,
) -> None:
    layers = np.arange(N_LAYERS)
    for task in tasks:
        group = task_group(task)
        ax.plot(layers, data[task], color=GROUP_COLORS[group], lw=0.7, alpha=0.2)

    grouped = group_curves(data, tasks)
    for group in GROUPS:
        if group not in grouped:
            continue
        arr = grouped[group]
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0)
        label = f"{GROUP_LABELS[group]} (n={arr.shape[0]})"
        ax.plot(layers, mean, color=GROUP_COLORS[group], lw=2.3, label=label)
        ax.fill_between(layers, mean - sd, mean + sd, color=GROUP_COLORS[group], alpha=0.13)

    if reference_y is not None:
        ax.axhline(reference_y, color="0.55", lw=0.9, ls=":")
    ax.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.35, zorder=-1)
    ax.set_title(title)
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.3, 11.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, axis="y", alpha=0.18)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def write_summary_csv(
    mean_probe: dict[str, np.ndarray],
    quantile_ratio: dict[str, np.ndarray],
    tasks: list[str],
) -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "group", "layer", "mean_probe_r2", "quantile_pinball_ratio"])
        for task in tasks:
            for layer in range(N_LAYERS):
                writer.writerow(
                    [
                        task,
                        task_group(task),
                        layer,
                        f"{mean_probe[task][layer]:.8g}",
                        f"{quantile_ratio[task][layer]:.8g}",
                    ]
                )


def main() -> None:
    mean_probe = load_mean_probe()
    quantile_ratio = load_quantile_pinball_ratio()
    tasks = [task for task in TASK_ORDER if task in mean_probe and task in quantile_ratio]
    if not tasks:
        raise SystemExit("No overlapping mean-probe and quantile-validation runs found.")

    missing_mean = sorted(set(TASK_ORDER) - set(mean_probe))
    missing_quantile = sorted(set(TASK_ORDER) - set(quantile_ratio))
    if missing_mean:
        print(f"Skipping tasks without mean probes: {', '.join(missing_mean)}")
    if missing_quantile:
        print(f"Skipping tasks without quantile validation: {', '.join(missing_quantile)}")

    write_summary_csv(mean_probe, quantile_ratio, tasks)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.45), constrained_layout=True)
    plot_grouped_curves(
        axes[0],
        mean_probe,
        tasks,
        ylabel=r"Ridge val $R^2$",
        title="A. Mean-parameter decoding",
        reference_y=0.0,
    )
    axes[0].set_ylim(-0.08, 1.02)

    plot_grouped_curves(
        axes[1],
        quantile_ratio,
        tasks,
        ylabel="Pinball ratio (probe / reference)",
        title="B. Quantile decoding",
        reference_y=1.0,
        log_y=True,
    )
    axes[1].set_ylim(0.9, 20.0)
    axes[1].set_yticks([1, 2, 3, 5, 10, 20])
    axes[1].set_yticklabels(["1", "2", "3", "5", "10", "20"])

    handles = [
        mlines.Line2D([], [], color=GROUP_COLORS[group], lw=2.3, label=GROUP_LABELS[group])
        for group in GROUPS
    ]
    axes[1].legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92)

    for suffix in ("png", "pdf"):
        out = FIG_DIR / f"probe_quantile_two_panel.{suffix}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")
    print(f"Wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
