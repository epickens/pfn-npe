"""Plotting utilities for SBI benchmarking.

All matplotlib code lives here — keeps the inference scripts clean.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_diagnostics(
    posterior_samples: np.ndarray,
    reference_samples: np.ndarray,
    history: dict,
    label: str,
    output_path: str | Path,
) -> None:
    """Training curves + posterior scatter + marginal histograms.

    Handles arbitrary theta dimensionality:
    - Scatter plot uses first 2 dims
    - Marginals show up to first 4 dims
    """
    dim_theta = posterior_samples.shape[1]
    n_marginals = min(dim_theta, 4)

    fig, axes = plt.subplots(1, 2 + n_marginals, figsize=(5 * (2 + n_marginals), 5))

    # ── Loss curves ──
    ax = axes[0]
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if train_loss:
        ax.plot(train_loss, label="Train", alpha=0.7)
    if val_loss:
        ax.plot(val_loss, label="Val", alpha=0.7)
    if train_loss or val_loss:
        ax.axvline(history.get("best_epoch", 0), color="red", ls="--", alpha=0.5, label="Best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood")
    ax.legend()
    ax.set_title("Training curves")

    # ── 2D scatter (first 2 dims) ──
    ax = axes[1]
    d0, d1 = 0, min(1, dim_theta - 1)
    ax.scatter(
        reference_samples[:2000, d0], reference_samples[:2000, d1],
        alpha=0.15, s=1, label="Reference", color="C0",
    )
    ax.scatter(
        posterior_samples[:2000, d0], posterior_samples[:2000, d1],
        alpha=0.15, s=1, label=label, color="C1",
    )
    ax.set_xlabel(f"theta_{d0 + 1}")
    ax.set_ylabel(f"theta_{d1 + 1}")
    ax.legend(markerscale=10)
    ax.set_title("Posterior samples")

    # ── Marginal histograms ──
    for j in range(n_marginals):
        ax = axes[2 + j]
        ax.hist(
            reference_samples[:, j], bins=50, alpha=0.4,
            density=True, label="Reference", color="C0",
        )
        ax.hist(
            posterior_samples[:, j], bins=50, alpha=0.4,
            density=True, label=label, color="C1",
        )
        ax.set_xlabel(f"theta_{j + 1}")
        ax.set_title(f"Marginal {j + 1}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def plot_loss_curves(
    history: dict,
    output_path: str | Path,
    title: str = "Training curves",
) -> None:
    """Standalone loss curve plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train", alpha=0.7)
    ax.plot(history["val_loss"], label="Val", alpha=0.7)
    ax.axvline(history.get("best_epoch", 0), color="red", ls="--", alpha=0.5, label="Best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood")
    ax.legend()
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bayesflow_summary(
    results_list: list[dict],
    output_path: str | Path,
) -> None:
    """Bar chart of BayesFlow C2ST scores across tasks."""
    task_names = [r["task_name"] for r in results_list]
    means = [np.mean(r["c2st_bayesflow"]) for r in results_list]
    stds = [np.std(r["c2st_bayesflow"]) for r in results_list]

    x = np.arange(len(task_names))

    fig, ax = plt.subplots(figsize=(max(8, len(task_names) * 2), 5))
    ax.bar(x, means, yerr=stds, color="C2", alpha=0.8, capsize=3, label="BayesFlow NPE")

    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Perfect (0.5)")
    ax.set_ylabel("C2ST (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.legend()
    ax.set_title("BayesFlow NPE Baseline")
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot: {output_path}")


def plot_summary_table(
    results_list: list[dict],
    output_path: str | Path,
) -> None:
    """Bar chart comparing methods across tasks."""
    task_names = [r["task_name"] for r in results_list]
    tabpfn_means = [np.mean(r["c2st_tabpfn"]) for r in results_list]
    tabpfn_stds = [np.std(r["c2st_tabpfn"]) for r in results_list]
    raw_means = [np.mean(r["c2st_raw"]) for r in results_list]
    raw_stds = [np.std(r["c2st_raw"]) for r in results_list]

    x = np.arange(len(task_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(task_names) * 2), 5))
    ax.bar(x - width / 2, tabpfn_means, width, yerr=tabpfn_stds,
           label="TabPFN embeddings", color="C0", alpha=0.8, capsize=3)
    ax.bar(x + width / 2, raw_means, width, yerr=raw_stds,
           label="Raw features", color="C1", alpha=0.8, capsize=3)

    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Perfect (0.5)")
    ax.set_ylabel("C2ST (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.legend()
    ax.set_title("TabPFN Embeddings vs Raw Features")
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot: {output_path}")


def plot_layer_sweep(
    sweep_results: dict[str, list[dict]],
    output_path: str | Path,
) -> None:
    """Line plot of C2ST vs layer index for each task.

    One subplot per task, with the raw-features baseline as a horizontal line.
    """
    task_names = list(sweep_results.keys())
    n_tasks = len(task_names)
    n_cols = min(n_tasks, 3)
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for i, task_name in enumerate(task_names):
        ax = axes[i // n_cols][i % n_cols]
        layer_results = sweep_results[task_name]

        layers = [r["layer"] for r in layer_results]
        means = [np.mean(r["c2st_tabpfn"]) for r in layer_results]
        stds = [np.std(r["c2st_tabpfn"]) for r in layer_results]
        raw_mean = np.mean(layer_results[0]["c2st_raw"])
        raw_std = np.std(layer_results[0]["c2st_raw"])

        ax.errorbar(layers, means, yerr=stds, marker="o", capsize=3,
                     label="TabPFN", color="C0", linewidth=2, markersize=5)
        ax.axhline(raw_mean, color="C1", ls="--", linewidth=1.5, label="Raw baseline")
        ax.fill_between(
            [min(layers) - 0.5, max(layers) + 0.5],
            raw_mean - raw_std, raw_mean + raw_std,
            color="C1", alpha=0.15,
        )
        ax.axhline(0.5, color="gray", ls=":", alpha=0.4, label="Perfect (0.5)")

        best_idx = int(np.argmin(means))
        ax.annotate(
            f"L{layers[best_idx]}",
            xy=(layers[best_idx], means[best_idx]),
            xytext=(0, -18), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold", color="C0",
        )

        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("C2ST")
        ax.set_title(task_name)
        ax.set_xticks(layers)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8, loc="upper right")

    # Hide unused subplots
    for j in range(n_tasks, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle("TabPFN Layer Sweep: C2ST vs Transformer Layer", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved layer sweep plot: {output_path}")


def plot_sbc(
    sbc_results: dict,
    label: str,
    output_path: str | Path,
) -> None:
    """Rank histogram diagnostic for Simulation-Based Calibration.

    One subplot per parameter dimension (up to 8). Well-calibrated posteriors
    produce uniform rank histograms; deviations indicate bias or miscalibration.

    Args:
        sbc_results: dict from compute_sbc with keys "ranks", "ks_stats", etc.
        label: method name for the title (e.g., "TabPFN emb", "Raw features")
        output_path: where to save the figure
    """
    ranks = sbc_results["ranks"]
    ks_stats = sbc_results["ks_stats"]
    n_trials, dim_theta = ranks.shape
    n_plots = min(dim_theta, 8)
    n_posterior = int(ranks.max()) + 1  # infer L from max rank

    n_cols = min(n_plots, 4)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    n_bins = min(20, n_posterior // 5)
    expected_count = n_trials / n_bins

    for j in range(n_plots):
        ax = axes[j // n_cols][j % n_cols]
        ax.hist(ranks[:, j], bins=n_bins, range=(0, n_posterior),
                color="C0", alpha=0.7, edgecolor="white", linewidth=0.5)

        # Expected uniform band (mean +/- 2 std of binomial)
        binom_std = np.sqrt(expected_count * (1 - 1 / n_bins))
        ax.axhline(expected_count, color="black", ls="--", linewidth=1, alpha=0.7)
        ax.axhspan(expected_count - 2 * binom_std, expected_count + 2 * binom_std,
                    color="gray", alpha=0.15)

        ax.set_xlabel("Rank")
        ax.set_ylabel("Count")
        ax.set_title(f"theta_{j + 1}  (KS={ks_stats[j]:.3f})")

    # Hide unused subplots
    for j in range(n_plots, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    mean_ks = sbc_results["mean_ks"]
    fig.suptitle(f"SBC Rank Histograms — {label}  (mean KS={mean_ks:.4f})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SBC plot: {output_path}")


def plot_budget_gap(
    budget_results: dict[str, dict[int, dict]],
    output_path: str | Path,
) -> None:
    """Line chart of C2ST gap (Raw - TabPFN) vs simulation budget.

    One line per task. Positive gap = TabPFN wins.

    Args:
        budget_results: {task_name: {budget: {"tabpfn": array, "raw": array}}}
        output_path: where to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for task_name, budgets_data in budget_results.items():
        budgets = sorted(budgets_data.keys())
        means = []
        stds = []
        for n in budgets:
            d = budgets_data[n]
            gaps = d["raw"] - d["tabpfn"]  # per-observation gaps
            means.append(np.mean(gaps))
            stds.append(np.std(gaps) / np.sqrt(len(gaps)))  # SEM

        ax.errorbar(budgets, means, yerr=stds, marker="o", capsize=3,
                     linewidth=1.5, markersize=4, label=task_name)

    ax.axhline(0, color="black", ls="-", linewidth=0.8, alpha=0.5)
    ax.fill_between(ax.get_xlim(), -0.02, 0.02, color="gray", alpha=0.1)
    ax.set_xscale("log")
    ax.set_xlabel("Simulation budget (n_train)")
    ax.set_ylabel("C2ST gap (Raw - TabPFN)\npositive = TabPFN wins")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.set_title("TabPFN Advantage vs Simulation Budget")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved budget gap plot: {output_path}")


def plot_budget_curves(
    budget_results: dict[str, dict[int, dict]],
    output_path: str | Path,
    methods: list[str] | None = None,
    method_labels: dict[str, str] | None = None,
    method_styles: dict[str, dict] | None = None,
    task_labels: dict[str, str] | None = None,
    title: str | None = "C2ST vs Simulation Budget",
    xlabel: str = "n_train",
    ylabel: str = "C2ST",
    ylim: tuple[float, float] | None = (0.45, 1.05),
    share_legend: bool = False,
    legend_ncols: int | None = None,
    n_cols: int | None = None,
    panel_size: tuple[float, float] = (5.0, 4.0),
    box_aspect: float | None = None,
    sharey: bool = True,
    xticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    y_label_first_col_only: bool = False,
    x_label_first_col_only: bool = False,
    x_label_bottom_row_only: bool = False,
    x_tick_bottom_row_only: bool = False,
    title_fontsize: float = 14.0,
    axis_labelsize: float = 10.0,
    tick_labelsize: float = 9.0,
    task_title_fontsize: float = 10.0,
    title_fontweight: str = "normal",
    axis_labelweight: str = "normal",
    tick_labelweight: str = "normal",
    task_title_fontweight: str = "normal",
    legend_fontsize: float = 9.0,
    line_alpha: float = 1.0,
    grid_alpha: float = 0.3,
    show_errorbars: bool = True,
    legend_bbox_y: float = 0.01,
    layout_rect: tuple[float, float, float, float] | None = None,
    subplot_wspace: float | None = None,
    subplot_hspace: float | None = None,
    subplot_left: float | None = None,
    subplot_right: float | None = None,
    subplot_bottom: float | None = None,
    subplot_top: float | None = None,
    suptitle_y: float | None = None,
    bbox_inches: str | None = "tight",
) -> None:
    """Multi-panel C2ST vs budget chart. One subplot per task, one line per method.

    Scales to arbitrary number of methods for baseline comparisons.
    Methods with data at only some budget points are plotted at those points only.

    Args:
        budget_results: {task_name: {budget: {"method1": array_or_scalar, ...}}}
        output_path: where to save the figure
        methods: list of method keys to plot (default: all keys across all entries)
        method_labels: {method_key: display_label} (default: use keys as labels)
        method_styles: {method_key: {"color": ..., "linestyle": ..., ...}}
        task_labels: {task_key: display_label} for subplot titles
        title: figure title; pass None to suppress
        xlabel, ylabel: axis labels
        ylim: y-axis limits; pass None for automatic limits
        share_legend: put a single legend under the whole figure
        legend_ncols: number of columns in shared legend
    """
    task_names = list(budget_results.keys())
    n_tasks = len(task_names)
    n_cols = n_cols if n_cols is not None else min(n_tasks, 4)
    n_cols = max(1, min(n_cols, n_tasks))
    n_rows = (n_tasks + n_cols - 1) // n_cols

    # Infer methods from all data points
    if methods is None:
        method_set: set[str] = set()
        for budgets_data in budget_results.values():
            for bdata in budgets_data.values():
                method_set.update(bdata.keys())
        methods = sorted(method_set)
    if method_labels is None:
        method_labels = {m: m for m in methods}
    if method_styles is None:
        method_styles = {}
    if task_labels is None:
        task_labels = {}

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        squeeze=False,
        sharey=sharey,
    )
    legend_handles = []
    legend_labels = []

    for i, task_name in enumerate(task_names):
        ax = axes[i // n_cols][i % n_cols]
        budgets_data = budget_results[task_name]
        all_budgets = sorted(budgets_data.keys())

        for ci, method in enumerate(methods):
            # Collect only budgets where this method has data
            plot_budgets = []
            means = []
            stds = []
            for n in all_budgets:
                if method not in budgets_data[n]:
                    continue
                scores = np.atleast_1d(budgets_data[n][method])
                plot_budgets.append(n)
                means.append(np.mean(scores))
                stds.append(np.std(scores))

            if not plot_budgets:
                continue

            style = method_styles.get(method, {})
            label = method_labels.get(method, method)
            if show_errorbars:
                container = ax.errorbar(
                    plot_budgets,
                    means,
                    yerr=stds,
                    marker=style.get("marker", "o"),
                    capsize=style.get("capsize", 3),
                    linewidth=style.get("linewidth", 1.5),
                    markersize=style.get("markersize", 4),
                    color=style.get("color", f"C{ci}"),
                    linestyle=style.get("linestyle", "-"),
                    alpha=style.get("alpha", line_alpha),
                    label=label,
                )
                legend_handle = container.lines[0]
            else:
                (line,) = ax.plot(
                    plot_budgets,
                    means,
                    marker=style.get("marker", "o"),
                    linewidth=style.get("linewidth", 1.5),
                    markersize=style.get("markersize", 4),
                    color=style.get("color", f"C{ci}"),
                    linestyle=style.get("linestyle", "-"),
                    alpha=style.get("alpha", line_alpha),
                    label=label,
                )
                legend_handle = line
            if share_legend and label not in legend_labels:
                legend_handles.append(legend_handle)
                legend_labels.append(label)

        ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
        ax.set_xscale("log")
        if xticks is not None:
            ax.set_xticks(xticks)
            if xtick_labels is not None:
                ax.set_xticklabels(xtick_labels)
        if xlim is not None:
            ax.set_xlim(*xlim)
        row_idx = i // n_cols
        is_bottom_row = row_idx == n_rows - 1
        show_xlabel = ((not x_label_first_col_only) or (i % n_cols == 0)) and (
            (not x_label_bottom_row_only) or is_bottom_row
        )
        if show_xlabel:
            ax.set_xlabel(xlabel, fontsize=axis_labelsize, fontweight=axis_labelweight)
        else:
            ax.set_xlabel("")
        if (not y_label_first_col_only) or (i % n_cols == 0):
            ax.set_ylabel(ylabel, fontsize=axis_labelsize, fontweight=axis_labelweight)
        elif sharey:
            ax.tick_params(labelleft=False)
        ax.set_title(
            task_labels.get(task_name, task_name),
            fontsize=task_title_fontsize,
            fontweight=task_title_fontweight,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        if box_aspect is not None:
            ax.set_box_aspect(box_aspect)
        ax.tick_params(axis="both", labelsize=tick_labelsize)
        if x_tick_bottom_row_only and not is_bottom_row:
            ax.tick_params(labelbottom=False)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight(tick_labelweight)
        if not share_legend:
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=grid_alpha)

    # Hide unused subplots
    for j in range(n_tasks, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    if title is not None:
        fig.suptitle(
            title,
            fontsize=title_fontsize,
            fontweight=title_fontweight,
            y=suptitle_y if suptitle_y is not None else (0.995 if share_legend else 1.02),
        )
    if share_legend and legend_handles:
        ncols = legend_ncols or min(len(legend_handles), 5)
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center", bbox_to_anchor=(0.5, legend_bbox_y),
            ncol=ncols, fontsize=legend_fontsize, frameon=False,
        )
    rect = layout_rect if layout_rect is not None else ((0, 0.06, 1, 0.95) if share_legend else None)
    plt.tight_layout(rect=rect)
    if any(
        value is not None
        for value in (
            subplot_wspace,
            subplot_hspace,
            subplot_left,
            subplot_right,
            subplot_bottom,
            subplot_top,
        )
    ):
        fig.subplots_adjust(
            left=subplot_left,
            right=subplot_right,
            bottom=subplot_bottom,
            top=subplot_top,
            wspace=subplot_wspace,
            hspace=subplot_hspace,
        )
    plt.savefig(output_path, dpi=150, bbox_inches=bbox_inches)
    plt.close()
    print(f"Saved budget curves plot: {output_path}")
