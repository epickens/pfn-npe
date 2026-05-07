"""Plot Pareto frontiers for the anytime/runtime SBI benchmark.

The plot uses cached cell JSON files from ``run_anytime_runtime_sweep.py``.
It treats lower runtime and lower posterior error as better, where posterior
error is ``2 * (joint C2ST - 0.5)`` by default.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_CELL_DIR = Path("pfn_testing/sbi/outputs/anytime_runtime/cells")
DEFAULT_OUT_DIR = Path("pfn_testing/sbi/outputs/anytime_runtime/figures")
DEFAULT_OUT_CSV = Path("pfn_testing/sbi/outputs/anytime_runtime/anytime_pareto_frontier.csv")
DEFAULT_SEED_OUT_CSV = Path("pfn_testing/sbi/outputs/anytime_runtime/anytime_seed_points_merged_npe_pfn.csv")
DEFAULT_TASKS = ["two_moons", "slcp", "ar1_ts_t50"]
DEFAULT_METHODS = [
    "pfn_npe_pca64",
    "npe_pfn_merged",
    "npe_pfn",
    "npe_pfn_e2",
    "npe_pfn_e4",
    "npe_pfn_e8",
    "sbi_npe_nsf",
    "sbi_fmpe",
]
SEED_PLOT_METHODS = [
    "pfn_npe_pca64",
    "npe_pfn_merged",
    "npe_pfn",
    "npe_pfn_e2",
    "npe_pfn_e4",
    "npe_pfn_e8",
    "sbi_npe_nsf",
    "sbi_fmpe",
]
METHOD_LABELS = {
    "pfn_npe_pca64": "PFN-NPE",
    "npe_pfn_merged": "NPE-PFN",
    "npe_pfn": "NPE-PFN x1",
    "npe_pfn_e2": "NPE-PFN x2",
    "npe_pfn_e4": "NPE-PFN x4",
    "npe_pfn_e8": "NPE-PFN x8",
    "sbi_npe": "sbi-NPE",
    "sbi_npe_nsf": "Raw-x NSF",
    "bayesflow_nsf_family": "BayesFlow",
    "bayesflow_nsf": "BayesFlow",
    "bayesflow_nsf_d6_b512_e80": "BayesFlow d6 fast",
    "bayesflow_nsf_d4_b512_e80": "BayesFlow d4 fast",
    "bayesflow_nsf_light": "BayesFlow light",
    "sbi_nle": "sbi-NLE",
    "sbi_nle_vi": "sbi-NLE VI",
    "sbi_nre": "sbi-NRE",
    "sbi_fmpe": "sbi-FMPE",
}
METHOD_COLORS = {
    "pfn_npe_pca64": "#0072B2",
    "npe_pfn_merged": "#D55E00",
    "npe_pfn": "#D55E00",
    "npe_pfn_e2": "#E69F00",
    "npe_pfn_e4": "#A6761D",
    "npe_pfn_e8": "#7F3B08",
    "sbi_npe": "#009E73",
    "sbi_npe_nsf": "#44AA99",
    "bayesflow_nsf_family": "#332288",
    "bayesflow_nsf": "#332288",
    "bayesflow_nsf_d6_b512_e80": "#5E3C99",
    "bayesflow_nsf_d4_b512_e80": "#8073AC",
    "bayesflow_nsf_light": "#B2ABD2",
    "sbi_nle": "#CC79A7",
    "sbi_nle_vi": "#AA4499",
    "sbi_nre": "#E69F00",
    "sbi_fmpe": "#CC6677",
}
TASK_LABELS = {
    "two_moons": "Two moons",
    "slcp": "SLCP",
    "ar1_ts_t50": "AR(1), T=50",
}

METRIC_LABELS = {
    "joint_c2st": "Joint C2ST error",
    "marginal_c2st": "Marginal C2ST error",
    "rank_c2st": "Rank C2ST error",
}

METHOD_FAMILY_MAP = {
    "npe_pfn": "npe_pfn_merged",
    "npe_pfn_e2": "npe_pfn_merged",
    "npe_pfn_e4": "npe_pfn_merged",
    "npe_pfn_e8": "npe_pfn_merged",
    "bayesflow_nsf": "bayesflow_nsf_family",
    "bayesflow_nsf_d6_b512_e80": "bayesflow_nsf_family",
    "bayesflow_nsf_d4_b512_e80": "bayesflow_nsf_family",
    "bayesflow_nsf_light": "bayesflow_nsf_family",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_cells(cell_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(cell_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["cell_file"] = path.name
        rows.append(row)
    if not rows:
        raise SystemExit(f"No cell JSON files found in {cell_dir}")
    df = pd.DataFrame(rows)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise SystemExit(f"No successful cell JSON files found in {cell_dir}")
    df = add_method_variants(df)
    return df


def add_method_variants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "npe_pfn_n_estimators" not in df.columns:
        df["npe_pfn_n_estimators"] = 1
    df["npe_pfn_n_estimators"] = df["npe_pfn_n_estimators"].fillna(1).astype(int)

    def clean_value(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        value_str = str(value)
        if value_str in {"", "None", "nan"}:
            return None
        return value_str

    def int_value(value: object) -> int | None:
        cleaned = clean_value(value)
        if cleaned is None:
            return None
        return int(float(cleaned))

    def tuple_value(value: object) -> tuple[int, ...]:
        if isinstance(value, list | tuple):
            return tuple(int(v) for v in value)
        cleaned = clean_value(value)
        if cleaned is None:
            return ()
        parsed = ast.literal_eval(cleaned)
        if not isinstance(parsed, list | tuple):
            return ()
        return tuple(int(v) for v in parsed)

    def meta_value(row: pd.Series, key: str) -> str | None:
        meta = row.get("method_meta", {})
        if not isinstance(meta, dict):
            return None
        return clean_value(meta.get(key))

    def method_variant(row: pd.Series) -> str:
        method = str(row["method"])
        if method == "npe_pfn":
            n_estimators = int(row["npe_pfn_n_estimators"])
            if n_estimators == 1:
                return method
            return f"npe_pfn_e{n_estimators}"
        if method == "sbi_npe":
            estimator = clean_value(row.get("sbi_density_estimator")) or meta_value(row, "estimator")
            if estimator == "nsf":
                return "sbi_npe_nsf"
            return method
        if method == "sbi_nle":
            sample_with = clean_value(row.get("sbi_sample_with")) or meta_value(row, "sbi_sample_with")
            if sample_with == "vi":
                return "sbi_nle_vi"
            return method
        if method == "bayesflow_nsf":
            depth = int_value(row.get("bayesflow_depth"))
            widths = tuple_value(row.get("bayesflow_widths"))
            bins = int_value(row.get("bayesflow_bins")) or 16
            epochs = int_value(row.get("bayesflow_max_epochs"))
            patience = int_value(row.get("bayesflow_patience"))
            batch_size = int_value(row.get("bayesflow_batch_size"))
            if (depth, widths, bins, epochs, patience, batch_size) == (6, (128, 128), 16, 80, 10, 512):
                return "bayesflow_nsf_d6_b512_e80"
            if (depth, widths, bins, epochs, patience, batch_size) == (4, (128, 128), 16, 80, 10, 512):
                return "bayesflow_nsf_d4_b512_e80"
            if (depth, widths, bins, epochs, patience, batch_size) == (4, (64, 64), 8, 60, 10, 1024):
                return "bayesflow_nsf_light"
            return method
        return method

    df["method"] = df.apply(method_variant, axis=1)
    return df


def aggregate_cells(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    needed = {
        "task",
        "method",
        "n_train",
        "seed",
        "n_ref",
        "n_posterior_samples",
        metric,
        "total_seconds_n_ref",
        "total_seconds_1_obs",
        "train_seconds",
        "sample_seconds_n_ref",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    group_cols = ["task", "method", "n_train", "n_ref", "n_posterior_samples"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
            total_seconds_mean=("total_seconds_n_ref", "mean"),
            total_seconds_std=("total_seconds_n_ref", "std"),
            total_seconds_1_obs_mean=("total_seconds_1_obs", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            sample_seconds_n_ref_mean=("sample_seconds_n_ref", "mean"),
        )
        .sort_values(["task", "method", "n_train"])
    )
    agg["metric_std"] = agg["metric_std"].fillna(0.0)
    agg["total_seconds_std"] = agg["total_seconds_std"].fillna(0.0)
    agg["c2st_error_mean"] = 2.0 * (agg["metric_mean"] - 0.5)
    agg["c2st_error_std"] = 2.0 * agg["metric_std"]
    agg["dominated"] = False
    agg["frontier"] = False

    for task, task_df in agg.groupby("task"):
        idx = pareto_indices(
            task_df["total_seconds_mean"].to_numpy(float),
            task_df["c2st_error_mean"].to_numpy(float),
        )
        task_indices = task_df.index.to_numpy()
        frontier_indices = task_indices[idx]
        agg.loc[frontier_indices, "frontier"] = True
        agg.loc[task_indices[~np.isin(np.arange(len(task_indices)), idx)], "dominated"] = True
    return agg


def pareto_indices(runtime: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Return non-dominated indices for lower runtime and lower error."""
    ok = np.isfinite(runtime) & np.isfinite(error) & (runtime > 0)
    candidate_indices = np.flatnonzero(ok)
    frontier: list[int] = []
    for i in candidate_indices:
        dominated = False
        for j in candidate_indices:
            if i == j:
                continue
            no_worse = runtime[j] <= runtime[i] and error[j] <= error[i]
            strictly_better = runtime[j] < runtime[i] or error[j] < error[i]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return np.asarray(frontier, dtype=int)


def recompute_global_dominance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dominated"] = False
    df["frontier"] = False
    for task, task_df in df.groupby("task"):
        idx = pareto_indices(
            task_df["total_seconds_mean"].to_numpy(float),
            task_df["c2st_error_mean"].to_numpy(float),
        )
        task_indices = task_df.index.to_numpy()
        frontier_indices = task_indices[idx]
        df.loc[frontier_indices, "frontier"] = True
        df.loc[task_indices[~np.isin(np.arange(len(task_indices)), idx)], "dominated"] = True
    return df


def method_family(method: object) -> str:
    method_str = str(method)
    return METHOD_FAMILY_MAP.get(method_str, method_str)


def collapse_aggregate_to_method_families(agg: pd.DataFrame, *, keep_dominated: bool) -> pd.DataFrame:
    """Collapse tuned variants into family labels without averaging variants."""
    collapsed = agg.copy()
    collapsed["source_method"] = collapsed["method"].astype(str)
    collapsed["method"] = collapsed["source_method"].map(method_family)
    if keep_dominated:
        return recompute_global_dominance(collapsed).sort_values(
            ["task", "method", "total_seconds_mean", "n_train"]
        )

    keep_indices: list[int] = []
    for _, family_df in collapsed.groupby(["task", "method"]):
        idx = pareto_indices(
            family_df["total_seconds_mean"].to_numpy(float),
            family_df["c2st_error_mean"].to_numpy(float),
        )
        keep_indices.extend(family_df.index.to_numpy()[idx].tolist())

    collapsed = collapsed.loc[sorted(keep_indices)].copy()
    collapsed = recompute_global_dominance(collapsed)
    return collapsed.sort_values(["task", "method", "total_seconds_mean", "n_train"])


def collapse_seed_points_to_method_families(df: pd.DataFrame) -> pd.DataFrame:
    collapsed = df.copy()
    collapsed["source_method"] = collapsed["method"].astype(str)
    collapsed["method"] = collapsed["source_method"].map(method_family)
    return collapsed


def filter_family_source_budgets(
    df: pd.DataFrame,
    keep_specs: dict[str, set[int]],
) -> pd.DataFrame:
    if not keep_specs or "source_method" not in df.columns:
        return df
    source = df["source_method"].astype(str)
    family = source.map(method_family)
    specified_families = {method_family(method) for method in keep_specs}
    keep = ~family.isin(specified_families)
    n_train = df["n_train"].astype(int)
    for method, budgets in keep_specs.items():
        keep |= (source == method) & n_train.isin(budgets)
    return df[keep].copy()


def filter_by_min_seeds(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    min_seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min_seeds <= 1:
        return df, agg
    group_cols = ["task", "method", "n_train", "n_ref", "n_posterior_samples"]
    keep_agg = agg["seed_count"].astype(int) >= min_seeds
    agg_filtered = agg[keep_agg].copy()
    if agg_filtered.empty:
        raise SystemExit(f"No aggregate cells have at least {min_seeds} seeds.")

    keep_keys = pd.MultiIndex.from_frame(agg_filtered[group_cols])
    df_keys = pd.MultiIndex.from_frame(df[group_cols])
    df_filtered = df[df_keys.isin(keep_keys)].copy()
    if df_filtered.empty:
        raise SystemExit(f"No seed-level cells remain after requiring at least {min_seeds} seeds.")
    return df_filtered, agg_filtered


def budget_label(n_train: int) -> str:
    if n_train >= 1000:
        value = n_train / 1000
        return f"{value:g}k"
    return str(n_train)


def ordered_values(values: pd.Series, preferred: list[str]) -> list[str]:
    present = set(values.astype(str))
    ordered = [value for value in preferred if value in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: tuple[str, ...]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def plot_frontier(
    agg: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
    metric: str,
    runtime_target: float | None,
    show_frontier: bool,
    label_all_points: bool,
    fade_dominated: bool,
    share_y: bool,
) -> None:
    configure_style()
    tasks = ordered_values(agg["task"], DEFAULT_TASKS)
    methods = ordered_values(agg["method"], DEFAULT_METHODS)
    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.35 * len(tasks), 2.85),
        squeeze=False,
        sharey=share_y,
    )
    global_y_max = max(0.1, float(agg["c2st_error_mean"].max()) + 0.07)

    for col, task in enumerate(tasks):
        ax = axes[0, col]
        task_df = agg[agg["task"] == task].copy()
        if runtime_target is not None:
            ax.axvline(runtime_target, color="0.35", lw=0.9, ls="--", zorder=0)
            ax.text(
                runtime_target,
                0.98,
                f"{runtime_target:g}s",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=6,
                color="0.35",
                rotation=90,
            )

        frontier = task_df[task_df["frontier"]].sort_values("total_seconds_mean")
        if show_frontier and len(frontier) > 1:
            ax.plot(
                frontier["total_seconds_mean"],
                frontier["c2st_error_mean"],
                color="0.05",
                lw=1.25,
                zorder=2,
                label="Pareto frontier" if col == 0 else None,
            )

        for method in methods:
            sub = task_df[task_df["method"] == method].sort_values("n_train")
            if sub.empty:
                continue
            color = METHOD_COLORS.get(method, "0.35")
            for _, row in sub.iterrows():
                dominated = bool(row["dominated"])
                faded = fade_dominated and dominated
                ax.scatter(
                    row["total_seconds_mean"],
                    row["c2st_error_mean"],
                    s=22 if faded else 32,
                    marker="o",
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.28 if faded else 0.95,
                    zorder=3 if faded else 4,
                    label=None,
                )
                if label_all_points:
                    ax.annotate(
                        budget_label(int(row["n_train"])),
                        (row["total_seconds_mean"], row["c2st_error_mean"]),
                        xytext=(2.5, 3.0),
                        textcoords="offset points",
                        fontsize=5.8,
                        color="0.45" if faded else "0.2",
                    )

        ax.axhline(0.0, color="0.45", lw=0.8, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Runtime for 10 posteriors (s)")
        if col == 0:
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        y_max = global_y_max if share_y else max(0.1, float(task_df["c2st_error_mean"].max()) + 0.07)
        ax.set_ylim(-0.02, y_max)

    legend_handles: list[Line2D] = []
    if show_frontier:
        legend_handles.append(Line2D([0], [0], color="0.05", lw=1.25, label="Pareto frontier"))
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
    legend_ncol = min(5, len(legend_handles))
    legend_rows = math.ceil(len(legend_handles) / legend_ncol) if legend_ncol else 1
    bottom = 0.12 + 0.045 * max(0, legend_rows - 1)
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="lower center", ncol=legend_ncol, frameon=False)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    save_figure(fig, out_dir, f"anytime_pareto_frontier_{metric}", formats)


def collapse_npe_pfn_methods(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_npe_pfn = df["method"].astype(str).str.startswith("npe_pfn")
    df.loc[is_npe_pfn, "method"] = "npe_pfn_merged"
    return df


def seed_point_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    needed = {
        "task",
        "method",
        "n_train",
        "seed",
        "n_ref",
        "n_posterior_samples",
        metric,
        "total_seconds_n_ref",
        "train_seconds",
        "sample_seconds_n_ref",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required seed plot columns: {missing}")
    columns = [
        "task",
        "method",
        "n_train",
        "seed",
        "n_ref",
        "n_posterior_samples",
        metric,
        "total_seconds_n_ref",
        "train_seconds",
        "sample_seconds_n_ref",
    ]
    if "source_method" in df.columns:
        columns.append("source_method")
    seed_df = df[columns].copy()
    seed_df["c2st_error"] = 2.0 * (seed_df[metric].astype(float) - 0.5)
    return seed_df.sort_values(["task", "method", "n_train", "seed"])


def plot_seed_points(
    seed_df: pd.DataFrame,
    agg: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
    metric: str,
    runtime_target: float | None,
    share_y: bool,
) -> None:
    configure_style()
    tasks = ordered_values(seed_df["task"], DEFAULT_TASKS)
    preferred_methods = SEED_PLOT_METHODS if "npe_pfn_merged" in set(seed_df["method"]) else DEFAULT_METHODS
    methods = ordered_values(seed_df["method"], preferred_methods)
    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.35 * len(tasks), 2.85),
        squeeze=False,
        sharey=share_y,
    )
    global_y_max = max(0.1, float(seed_df["c2st_error"].max()) + 0.07)

    for col, task in enumerate(tasks):
        ax = axes[0, col]
        task_seed_df = seed_df[seed_df["task"] == task].copy()
        task_agg = agg[agg["task"] == task].copy()
        if runtime_target is not None:
            ax.axvline(runtime_target, color="0.35", lw=0.9, ls="--", zorder=0)

        for method in methods:
            seed_sub = task_seed_df[task_seed_df["method"] == method]
            if seed_sub.empty:
                continue
            color = METHOD_COLORS.get(method, "0.35")
            ax.scatter(
                seed_sub["total_seconds_n_ref"],
                seed_sub["c2st_error"],
                s=18,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidth=0.65,
                alpha=0.42,
                zorder=2,
            )

            mean_sub = task_agg[task_agg["method"] == method].sort_values("n_train")
            if mean_sub.empty:
                continue
            ax.scatter(
                mean_sub["total_seconds_mean"],
                mean_sub["c2st_error_mean"],
                s=36,
                marker="o",
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.98,
                zorder=4,
            )

        ax.axhline(0.0, color="0.45", lw=0.8, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Runtime for 10 posteriors (s)")
        if col == 0:
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        y_max = global_y_max if share_y else max(0.1, float(task_seed_df["c2st_error"].max()) + 0.07)
        ax.set_ylim(-0.02, y_max)

    legend_handles = [
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
        for method in methods
        if method in set(seed_df["method"])
    ]
    legend_ncol = min(5, len(legend_handles))
    legend_rows = math.ceil(len(legend_handles) / legend_ncol) if legend_ncol else 1
    bottom = 0.12 + 0.045 * max(0, legend_rows - 1)
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="lower center", ncol=legend_ncol, frameon=False)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    save_figure(fig, out_dir, f"anytime_pareto_frontier_{metric}_all_seeds", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-dir", type=Path, default=DEFAULT_CELL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--seed-out-csv", type=Path, default=DEFAULT_SEED_OUT_CSV)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--metric", choices=sorted(METRIC_LABELS), default="joint_c2st")
    parser.add_argument("--n-ref", type=int, default=10)
    parser.add_argument("--n-posterior-samples", type=int, default=1000)
    parser.add_argument("--max-n-train", type=int, default=None)
    parser.add_argument(
        "--include-extra-budget",
        nargs="*",
        default=[],
        help=(
            "Extra method:budget entries to keep even when --max-n-train "
            "would exclude them. Budgets can be comma-separated, e.g. "
            "pfn_npe_pca64:50000,100000."
        ),
    )
    parser.add_argument("--runtime-target", type=float, default=None)
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help="Drop aggregate configurations with fewer successful seeds.",
    )
    parser.add_argument(
        "--share-y",
        action="store_true",
        help="Use the same y-axis limits across task panels.",
    )
    parser.add_argument("--show-frontier", action="store_true")
    parser.add_argument("--label-all-points", action="store_true")
    parser.add_argument("--fade-dominated", action="store_true")
    parser.add_argument("--plot-seeds", action="store_true")
    parser.add_argument("--merge-npe-pfn-seeds", action="store_true")
    parser.add_argument(
        "--collapse-method-families",
        action="store_true",
        help=(
            "Map tuned variants to method families and keep the within-family "
            "non-dominated configurations for the main plot."
        ),
    )
    parser.add_argument(
        "--keep-family-dominated",
        action="store_true",
        help="When collapsing method families, retain dominated variant points under the family label.",
    )
    parser.add_argument(
        "--keep-family-source-budget",
        nargs="*",
        default=[],
        help=(
            "When collapsing method families, retain only selected source "
            "method:budget entries inside any specified family. Budgets can "
            "be comma-separated, e.g. npe_pfn:200,3500 npe_pfn_e2:500."
        ),
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return parser.parse_args()


def parse_extra_budget_specs(specs: list[str]) -> set[tuple[str, int]]:
    extra: set[tuple[str, int]] = set()
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"Invalid --include-extra-budget entry {spec!r}; expected method:budget")
        method, budget_spec = spec.split(":", 1)
        method = method.strip()
        if not method:
            raise SystemExit(f"Invalid --include-extra-budget entry {spec!r}; method is empty")
        for budget in budget_spec.split(","):
            budget = budget.strip()
            if not budget:
                continue
            extra.add((method, int(budget)))
    return extra


def parse_source_budget_specs(specs: list[str]) -> dict[str, set[int]]:
    keep: dict[str, set[int]] = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"Invalid --keep-family-source-budget entry {spec!r}; expected method:budget")
        method, budget_spec = spec.split(":", 1)
        method = method.strip()
        if not method:
            raise SystemExit(f"Invalid --keep-family-source-budget entry {spec!r}; method is empty")
        budgets = keep.setdefault(method, set())
        for budget in budget_spec.split(","):
            budget = budget.strip()
            if not budget:
                continue
            budgets.add(int(budget))
    return keep


def main() -> None:
    args = parse_args()
    extra_budgets = parse_extra_budget_specs(args.include_extra_budget)
    family_keep_specs = parse_source_budget_specs(args.keep_family_source_budget)
    df = load_cells(args.cell_dir)
    df = df[
        df["task"].isin(args.tasks)
        & (df["n_ref"].astype(int) == args.n_ref)
        & (df["n_posterior_samples"].astype(int) == args.n_posterior_samples)
    ].copy()
    if args.max_n_train is not None:
        n_train = df["n_train"].astype(int)
        keep = n_train <= args.max_n_train
        if extra_budgets:
            extra_keep = pd.Series(False, index=df.index)
            for method, budget in extra_budgets:
                extra_keep |= (df["method"].astype(str) == method) & (n_train == budget)
            keep |= extra_keep
        df = df[keep].copy()
    if args.methods:
        df = df[df["method"].isin(args.methods)].copy()
    if df.empty:
        raise SystemExit("No cells left after filtering.")

    agg = aggregate_cells(df, args.metric)
    df, agg = filter_by_min_seeds(df, agg, args.min_seeds)
    if args.collapse_method_families:
        seed_input = collapse_seed_points_to_method_families(df)
        agg = collapse_aggregate_to_method_families(agg, keep_dominated=args.keep_family_dominated)
        if family_keep_specs:
            seed_input = filter_family_source_budgets(seed_input, family_keep_specs)
            agg = filter_family_source_budgets(agg, family_keep_specs)
            agg = recompute_global_dominance(agg)
    else:
        seed_input = collapse_npe_pfn_methods(df) if args.merge_npe_pfn_seeds else df
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(agg)} rows)")
    plot_frontier(
        agg,
        args.out_dir,
        tuple(args.formats),
        args.metric,
        args.runtime_target,
        args.show_frontier,
        args.label_all_points,
        args.fade_dominated,
        args.share_y,
    )
    if args.plot_seeds:
        seed_df = seed_point_table(seed_input, args.metric)
        seed_agg = agg if args.collapse_method_families else aggregate_cells(seed_input, args.metric)
        args.seed_out_csv.parent.mkdir(parents=True, exist_ok=True)
        seed_df.to_csv(args.seed_out_csv, index=False)
        print(f"Wrote {args.seed_out_csv} ({len(seed_df)} rows)")
        plot_seed_points(
            seed_df,
            seed_agg,
            args.out_dir,
            tuple(args.formats),
            args.metric,
            args.runtime_target,
            args.share_y,
        )

    print("\nPareto frontier cells:")
    for _, row in agg[agg["frontier"]].sort_values(["task", "total_seconds_mean"]).iterrows():
        print(
            f"  {row['task']:<12} {row['method']:<14} n={int(row['n_train']):<6} "
            f"error={row['c2st_error_mean']:.3f} joint={row['metric_mean']:.3f} "
            f"time={row['total_seconds_mean']:.1f}s"
        )


if __name__ == "__main__":
    main()
