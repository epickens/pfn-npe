"""Create the combined appendix view for anytime/runtime Pareto results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from plot_anytime_pareto_frontier import (
    DEFAULT_TASKS,
    METHOD_COLORS,
    METHOD_LABELS,
    SEED_PLOT_METHODS,
    TASK_LABELS,
    configure_style,
    ordered_values,
    save_figure,
)


DEFAULT_AGG_CSV = Path("pfn_testing/sbi/outputs/anytime_runtime/anytime_pareto_frontier_all_tasks_min3.csv")
DEFAULT_SEED_CSV = Path("pfn_testing/sbi/outputs/anytime_runtime/anytime_seed_points_all_tasks_min3.csv")
DEFAULT_OUT_DIR = Path("pfn_testing/sbi/outputs/anytime_runtime/figures/all_tasks_min3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg-csv", type=Path, default=DEFAULT_AGG_CSV)
    parser.add_argument("--seed-csv", type=Path, default=DEFAULT_SEED_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return parser.parse_args()


def set_task_axes(ax: plt.Axes, task: str, agg: pd.DataFrame, seed: pd.DataFrame) -> None:
    task_agg = agg[agg["task"] == task]
    task_seed = seed[seed["task"] == task]
    runtime = pd.concat(
        [
            task_agg["total_seconds_mean"].astype(float),
            task_seed["total_seconds_n_ref"].astype(float),
        ],
        ignore_index=True,
    )
    runtime = runtime[runtime > 0]
    if not runtime.empty:
        ax.set_xlim(float(runtime.min()) * 0.75, float(runtime.max()) * 1.35)
    ax.axhline(0.0, color="0.45", lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log")


def plot_combined(agg: pd.DataFrame, seed: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> None:
    configure_style()
    tasks = ordered_values(agg["task"], DEFAULT_TASKS)
    methods = ordered_values(agg["method"], SEED_PLOT_METHODS)
    fig, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(6.7, 7.4),
        squeeze=False,
        sharey=True,
    )
    y_max = max(0.1, float(seed["c2st_error"].max()) + 0.07)

    for row, task in enumerate(tasks):
        mean_ax, seed_ax = axes[row]
        task_agg = agg[agg["task"] == task].copy()
        task_seed = seed[seed["task"] == task].copy()

        if row == 0:
            mean_ax.set_title("Seed means")
            seed_ax.set_title("Individual seeds")

        frontier = task_agg[task_agg["frontier"].astype(bool)].sort_values("total_seconds_mean")
        if len(frontier) > 1:
            mean_ax.plot(
                frontier["total_seconds_mean"],
                frontier["c2st_error_mean"],
                color="0.05",
                lw=1.1,
                zorder=2,
            )

        for method in methods:
            color = METHOD_COLORS.get(method, "0.35")
            mean_sub = task_agg[task_agg["method"] == method].sort_values("n_train")
            for _, point in mean_sub.iterrows():
                dominated = bool(point["dominated"])
                mean_ax.scatter(
                    point["total_seconds_mean"],
                    point["c2st_error_mean"],
                    s=20 if dominated else 30,
                    marker="o",
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.28 if dominated else 0.95,
                    zorder=3 if dominated else 4,
                )

            seed_sub = task_seed[task_seed["method"] == method]
            if not seed_sub.empty:
                seed_ax.scatter(
                    seed_sub["total_seconds_n_ref"],
                    seed_sub["c2st_error"],
                    s=16,
                    marker="o",
                    facecolors="none",
                    edgecolors=color,
                    linewidth=0.6,
                    alpha=0.38,
                    zorder=2,
                )
            if not mean_sub.empty:
                seed_ax.scatter(
                    mean_sub["total_seconds_mean"],
                    mean_sub["c2st_error_mean"],
                    s=30,
                    marker="o",
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.98,
                    zorder=4,
                )

        for ax in (mean_ax, seed_ax):
            set_task_axes(ax, task, agg, seed)
            ax.set_ylim(-0.02, y_max)
            ax.text(
                0.97,
                0.90,
                TASK_LABELS.get(task, task),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                fontweight="bold",
            )
            if row == len(tasks) - 1:
                ax.set_xlabel("Runtime for 10 posteriors (s)")
        mean_ax.set_ylabel("Joint C2ST error")

    legend_handles: list[Line2D] = [
        Line2D([0], [0], color="0.05", lw=1.1, label="Pareto frontier"),
    ]
    for method in methods:
        if method not in set(agg["method"]):
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=METHOD_COLORS.get(method, "0.35"),
                markerfacecolor=METHOD_COLORS.get(method, "0.35"),
                markeredgecolor="white",
                lw=0,
                linestyle="None",
                label=METHOD_LABELS.get(method, method),
            )
        )
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=min(5, len(legend_handles)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save_figure(fig, out_dir, "anytime_pareto_frontier_joint_c2st_combined", formats)


def main() -> None:
    args = parse_args()
    agg = pd.read_csv(args.agg_csv)
    seed = pd.read_csv(args.seed_csv)
    plot_combined(agg, seed, args.out_dir, tuple(args.formats))


if __name__ == "__main__":
    main()
