"""Plot how posterior structure relates to TabPFN-NPE PCA64 performance.

This script consumes the post-hoc probe produced by
``posterior_error_variance_probe.py`` and the matching posterior samples in
``flow_vs_quantile``. It creates review figures that separate:

* intrinsic posterior structure,
* model-induced posterior moment/quantile errors,
* C2ST degradation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task  # noqa: E402


DEFAULT_PROBE = Path(
    "pfn_testing/sbi/outputs/layer_ablation/tabpfn_npe_pca64_n10000_variance_probe.csv"
)
DEFAULT_FLOW_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
DEFAULT_OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")

TASK_ORDER = ["two_moons", "ar1_ts_t50", "slcp"]
TASK_LABEL = {
    "two_moons": "Two moons",
    "ar1_ts_t50": "AR(1), T=50",
    "slcp": "SLCP",
}
TASK_COLOR = {
    "two_moons": "#0072B2",
    "ar1_ts_t50": "#009E73",
    "slcp": "#D55E00",
}
METHOD = "nsf_pca64"
_REFERENCE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def configure_style() -> None:
    mpl.rcParams.update(
        {
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
        }
    )


def load_probe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["task"].isin(TASK_ORDER)].copy()
    df["task_label"] = pd.Categorical(
        df["task"].map(TASK_LABEL),
        categories=[TASK_LABEL[t] for t in TASK_ORDER],
        ordered=True,
    )
    df["joint_gap"] = df["joint_c2st"] - df["marginal_c2st"]
    df["rank_gap"] = df["rank_c2st"] - df["marginal_c2st"]
    df["ref_marginal_var_sd"] = df["ref_var_anisotropy_cv"] * df["ref_total_var"]
    return df


def save_figure(
    fig: plt.Figure, out_dir: Path, stem: str, formats: tuple[str, ...]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def scatter_by_task(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str = "Joint C2ST",
    xscale: str | None = None,
    fit: bool = True,
) -> None:
    for task in TASK_ORDER:
        sub = df[df["task"] == task]
        ax.scatter(
            sub[x],
            sub[y],
            s=28,
            color=TASK_COLOR[task],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.86,
            label=TASK_LABEL[task],
        )
    if fit:
        xs = df[x].to_numpy(float)
        ys = df[y].to_numpy(float)
        mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0 if xscale == "log" else True)
        if mask.sum() >= 3:
            xfit = np.log10(xs[mask]) if xscale == "log" else xs[mask]
            coef = np.polyfit(xfit, ys[mask], deg=1)
            grid = np.linspace(xfit.min(), xfit.max(), 100)
            xplot = 10**grid if xscale == "log" else grid
            ax.plot(xplot, coef[0] * grid + coef[1], color="0.2", lw=1.2, alpha=0.8)
            rho, pval = spearmanr(xs[mask], ys[mask])
            ax.text(
                0.04,
                0.96,
                rf"$\rho$={rho:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 2.0},
            )
            if pval > 0.05:
                ax.text(0.04, 0.84, f"p={pval:.2g}", transform=ax.transAxes, fontsize=7)
    if xscale:
        ax.set_xscale(xscale)
    ax.axhline(0.5, color="0.45", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.48, max(0.9, float(df[y].max()) + 0.03))


def seed_sem(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def aggregate_observations_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, obs), sub in df.groupby(["task", "obs"], observed=True):
        row: dict[str, float | int | str] = {
            "task": str(task),
            "obs": int(obs),
            "n_seed": int(sub["seed"].nunique()),
        }
        for col in [
            "ref_var_anisotropy_cv",
            "ref_marginal_var_sd",
            "mean_rmse_std",
            "joint_c2st",
        ]:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_sem"] = seed_sem(sub[col])
        rows.append(row)

    out = pd.DataFrame(rows)
    out["task_label"] = pd.Categorical(
        out["task"].map(TASK_LABEL),
        categories=[TASK_LABEL[t] for t in TASK_ORDER],
        ordered=True,
    )
    return out.sort_values(["task_label", "obs"])


def add_reference_geometry_score(df: pd.DataFrame) -> pd.DataFrame:
    unique_obs = df[["task", "obs", "ref_var_anisotropy_cv"]].drop_duplicates().copy()
    unique_obs["ref_geometry_score"] = unique_obs["ref_var_anisotropy_cv"].rank(
        method="average",
        pct=True,
    )
    return df.merge(
        unique_obs[["task", "obs", "ref_geometry_score"]], on=["task", "obs"]
    )


def log_safe_xerr(x: np.ndarray, xerr: np.ndarray) -> np.ndarray:
    lower = np.minimum(xerr, np.maximum(0.0, x * 0.9))
    return np.vstack([lower, xerr])


def errorbar_by_task(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    yerr: str,
    xlabel: str,
    *,
    xerr: str | None = None,
    ylabel: str = "Mean joint C2ST across seeds",
    xscale: str | None = None,
    fit: bool = True,
) -> None:
    for task in TASK_ORDER:
        sub = df[df["task"] == task]
        xs = sub[x].to_numpy(dtype=np.float64)
        ys = sub[y].to_numpy(dtype=np.float64)
        yerrs = sub[yerr].to_numpy(dtype=np.float64)
        xerrs = None
        if xerr is not None:
            raw_xerr = sub[xerr].to_numpy(dtype=np.float64)
            xerrs = log_safe_xerr(xs, raw_xerr) if xscale == "log" else raw_xerr

        ax.errorbar(
            xs,
            ys,
            xerr=xerrs,
            yerr=yerrs,
            fmt="o",
            ms=5.2,
            color=TASK_COLOR[task],
            ecolor=TASK_COLOR[task],
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            alpha=0.88,
            label=TASK_LABEL[task],
        )

    xs = df[x].to_numpy(dtype=np.float64)
    ys = df[y].to_numpy(dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if xscale == "log":
        mask &= xs > 0
    if fit and mask.sum() >= 3:
        xfit = np.log10(xs[mask]) if xscale == "log" else xs[mask]
        coef = np.polyfit(xfit, ys[mask], deg=1)
        grid = np.linspace(xfit.min(), xfit.max(), 100)
        xplot = 10**grid if xscale == "log" else grid
        ax.plot(xplot, coef[0] * grid + coef[1], color="0.2", lw=1.1, alpha=0.8)
        rho, _ = spearmanr(xs[mask], ys[mask])
        ax.text(
            0.04,
            0.96,
            rf"$\rho$={rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 2.0},
        )
    if xscale:
        ax.set_xscale(xscale)
    ax.axhline(0.5, color="0.45", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.48, max(0.9, float(df[y].max()) + float(df[yerr].max()) + 0.03))


def seed_markers(df: pd.DataFrame) -> dict[int, str]:
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    seeds = sorted(int(seed) for seed in df["seed"].unique())
    return {seed: markers[i % len(markers)] for i, seed in enumerate(seeds)}


def scatter_by_task_and_seed(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    marker_by_seed: dict[int, str],
    *,
    ylabel: str = "Joint C2ST",
    xscale: str | None = None,
    fit: bool = True,
) -> None:
    for task in TASK_ORDER:
        for seed, marker in marker_by_seed.items():
            sub = df[(df["task"] == task) & (df["seed"] == seed)]
            ax.scatter(
                sub[x],
                sub[y],
                s=32,
                marker=marker,
                color=TASK_COLOR[task],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
            )

    xs = df[x].to_numpy(dtype=np.float64)
    ys = df[y].to_numpy(dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if xscale == "log":
        mask &= xs > 0
    if fit and mask.sum() >= 3:
        xfit = np.log10(xs[mask]) if xscale == "log" else xs[mask]
        coef = np.polyfit(xfit, ys[mask], deg=1)
        grid = np.linspace(xfit.min(), xfit.max(), 100)
        xplot = 10**grid if xscale == "log" else grid
        ax.plot(xplot, coef[0] * grid + coef[1], color="0.2", lw=1.1, alpha=0.8)
        rho, _ = spearmanr(xs[mask], ys[mask])
        ax.text(
            0.04,
            0.96,
            rf"$\rho$={rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 2.0},
        )

    if xscale:
        ax.set_xscale(xscale)
    ax.axhline(0.5, color="0.45", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.48, max(0.9, float(df[y].max()) + 0.03))


def plot_seed_level_structure_location(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    marker_by_seed = seed_markers(df)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.25), sharey=True)

    scatter_by_task_and_seed(
        axes[0],
        df,
        "ref_var_anisotropy_cv",
        "joint_c2st",
        "Reference posterior anisotropy",
        marker_by_seed,
    )
    axes[0].set_title("Intrinsic structure")
    panel_label(axes[0], "A")

    scatter_by_task_and_seed(
        axes[1],
        df,
        "mean_rmse_std",
        "joint_c2st",
        "Mean error / posterior s.d.",
        marker_by_seed,
        xscale="log",
    )
    axes[1].set_title("Location error")
    axes[1].set_ylabel("")
    panel_label(axes[1], "B")

    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TASK_COLOR[task],
            markeredgecolor="white",
            markersize=6,
            label=TASK_LABEL[task],
        )
        for task in TASK_ORDER
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="0.3",
            markerfacecolor="white",
            markeredgecolor="0.3",
            linestyle="none",
            markersize=5.5,
            label=str(seed),
        )
        for seed, marker in marker_by_seed.items()
    ]

    fig.legend(
        task_handles,
        [handle.get_label() for handle in task_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.39, -0.015),
    )
    fig.legend(
        seed_handles,
        [handle.get_label() for handle in seed_handles],
        loc="lower center",
        ncol=len(seed_handles),
        frameon=False,
        title="seed",
        bbox_to_anchor=(0.82, -0.015),
    )
    fig.suptitle(
        "Seed-level observations: structure and location error explain C2ST",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.98))
    save_figure(fig, out_dir, "tabpfn_seed_level_structure_location", formats)


def plot_reference_geometry_score(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    score_df = add_reference_geometry_score(df)
    marker_by_seed = seed_markers(score_df)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    scatter_by_task_and_seed(
        ax,
        score_df,
        "ref_geometry_score",
        "joint_c2st",
        "Reference geometry score\n(posterior anisotropy percentile)",
        marker_by_seed,
    )
    ax.set_xlim(0.0, 1.04)
    ax.set_title("Reference-only geometry vs joint C2ST")

    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TASK_COLOR[task],
            markeredgecolor="white",
            markersize=6,
            label=TASK_LABEL[task],
        )
        for task in TASK_ORDER
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="0.3",
            markerfacecolor="white",
            markeredgecolor="0.3",
            linestyle="none",
            markersize=5.5,
            label=str(seed),
        )
        for seed, marker in marker_by_seed.items()
    ]
    handles = task_handles + seed_handles
    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, frameon=False, ncol=2, loc="lower right")

    fig.tight_layout()
    save_figure(fig, out_dir, "tabpfn_reference_geometry_score", formats)


def plot_reference_anisotropy(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    marker_by_seed = seed_markers(df)

    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    scatter_by_task_and_seed(
        ax,
        df,
        "ref_var_anisotropy_cv",
        "joint_c2st",
        "Reference posterior anisotropy\nCV of marginal variances",
        marker_by_seed,
    )
    ax.set_title("Reference posterior anisotropy vs joint C2ST")

    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TASK_COLOR[task],
            markeredgecolor="white",
            markersize=6,
            label=TASK_LABEL[task],
        )
        for task in TASK_ORDER
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="0.3",
            markerfacecolor="white",
            markeredgecolor="0.3",
            linestyle="none",
            markersize=5.5,
            label=str(seed),
        )
        for seed, marker in marker_by_seed.items()
    ]
    handles = task_handles + seed_handles
    ax.legend(
        handles, [handle.get_label() for handle in handles], frameon=False, ncol=2
    )

    fig.tight_layout()
    save_figure(fig, out_dir, "tabpfn_reference_anisotropy", formats)


def plot_reference_anisotropy_seed_averaged(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    obs_df = aggregate_observations_over_seeds(df)

    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    errorbar_by_task(
        ax,
        obs_df,
        "ref_var_anisotropy_cv_mean",
        "joint_c2st_mean",
        "joint_c2st_sem",
        "Reference posterior anisotropy\nCV of marginal variances",
        ylabel="Mean joint C2ST across seeds",
    )
    ax.set_title("Reference posterior anisotropy vs mean joint C2ST")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save_figure(fig, out_dir, "tabpfn_reference_anisotropy_seed_averaged", formats)


def plot_reference_marginal_variance_sd(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    marker_by_seed = seed_markers(df)

    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    scatter_by_task_and_seed(
        ax,
        df,
        "ref_marginal_var_sd",
        "joint_c2st",
        "SD of reference marginal variances",
        marker_by_seed,
    )
    ax.set_title("Reference marginal-variance SD vs joint C2ST")

    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TASK_COLOR[task],
            markeredgecolor="white",
            markersize=6,
            label=TASK_LABEL[task],
        )
        for task in TASK_ORDER
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="0.3",
            markerfacecolor="white",
            markeredgecolor="0.3",
            linestyle="none",
            markersize=5.5,
            label=str(seed),
        )
        for seed, marker in marker_by_seed.items()
    ]
    handles = task_handles + seed_handles
    ax.legend(
        handles, [handle.get_label() for handle in handles], frameon=False, ncol=2
    )

    fig.tight_layout()
    save_figure(fig, out_dir, "tabpfn_reference_marginal_variance_sd", formats)


def plot_reference_marginal_variance_sd_seed_averaged(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    obs_df = aggregate_observations_over_seeds(df)

    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    errorbar_by_task(
        ax,
        obs_df,
        "ref_marginal_var_sd_mean",
        "joint_c2st_mean",
        "joint_c2st_sem",
        "SD of reference marginal variances",
        ylabel="Mean joint C2ST across seeds",
    )
    ax.set_title("Reference marginal-variance SD vs mean joint C2ST")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save_figure(
        fig, out_dir, "tabpfn_reference_marginal_variance_sd_seed_averaged", formats
    )


def plot_reference_corr_rank_gap(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    gap_df = add_reference_corr_rank_gap(df)
    marker_by_seed = seed_markers(gap_df)

    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    scatter_by_task_and_seed(
        ax,
        gap_df,
        "ref_corr_rank_gap",
        "joint_c2st",
        "Reference |Pearson corr - rank corr|\n(mean over parameter pairs)",
        marker_by_seed,
    )
    ax.set_title("Reference nonlinear dependence vs joint C2ST")

    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TASK_COLOR[task],
            markeredgecolor="white",
            markersize=6,
            label=TASK_LABEL[task],
        )
        for task in TASK_ORDER
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="0.3",
            markerfacecolor="white",
            markeredgecolor="0.3",
            linestyle="none",
            markersize=5.5,
            label=str(seed),
        )
        for seed, marker in marker_by_seed.items()
    ]
    handles = task_handles + seed_handles
    ax.legend(
        handles, [handle.get_label() for handle in handles], frameon=False, ncol=2
    )

    fig.tight_layout()
    save_figure(fig, out_dir, "tabpfn_reference_corr_rank_gap", formats)


def load_reference_descriptor_table(df: pd.DataFrame) -> pd.DataFrame | None:
    path = Path(
        "pfn_testing/sbi/outputs/layer_ablation/reference_posterior_descriptors_by_obs.csv"
    )
    if path.exists():
        return pd.read_csv(path)
    return None


def plot_reference_descriptor_sweep(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    descriptors = load_reference_descriptor_table(df)
    if descriptors is None:
        print("Skipping descriptor sweep: reference descriptor CSV not found.")
        return

    features = [
        ("mahalanobis_m2_skew", "Mahalanobis\nradius skew"),
        ("scale_imbalance_score", "Scale imbalance\nscore"),
        ("marginal_var_gini", "Marginal variance\nGini"),
        ("log_marginal_var_ratio", "Log max/min\nmarginal variance"),
        ("marginal_var_cv", "Marginal variance\nCV"),
        ("max_tail_975_iqr_ratio", "Max 95% width\n/ IQR"),
        ("mean_abs_marginal_skew", "Mean |marginal\nskew|"),
        ("gaussian_log_volume_per_dim", "Gaussian log volume\nper dim"),
    ]
    plot_df = df.merge(
        descriptors[["task", "obs", *(feature for feature, _ in features)]],
        on=["task", "obs"],
        how="left",
    )
    marker_by_seed = seed_markers(plot_df)

    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.2), sharey=True)
    for ax, (feature, label) in zip(axes.ravel(), features, strict=True):
        scatter_by_task_and_seed(
            ax,
            plot_df,
            feature,
            "joint_c2st",
            label,
            marker_by_seed,
        )
        ax.set_title(feature, fontsize=8)
    for ax in axes[:, 1:].ravel():
        ax.set_ylabel("")

    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TASK_COLOR[task],
            markeredgecolor="white",
            markersize=6,
            label=TASK_LABEL[task],
        )
        for task in TASK_ORDER
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="0.3",
            markerfacecolor="white",
            markeredgecolor="0.3",
            linestyle="none",
            markersize=5.5,
            label=str(seed),
        )
        for seed, marker in marker_by_seed.items()
    ]
    fig.legend(
        task_handles,
        [handle.get_label() for handle in task_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.38, -0.01),
    )
    fig.legend(
        seed_handles,
        [handle.get_label() for handle in seed_handles],
        loc="lower center",
        ncol=len(seed_handles),
        frameon=False,
        title="seed",
        bbox_to_anchor=(0.78, -0.01),
    )
    fig.suptitle(
        "Temporary descriptor sweep: reference posterior geometry vs joint C2ST", y=1.02
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save_figure(fig, out_dir, "tmp_reference_descriptor_sweep_seed_level", formats)

    obs_df = (
        plot_df.groupby(["task", "obs"], as_index=False)
        .agg(
            joint_c2st=("joint_c2st", "mean"),
            **{feature: (feature, "first") for feature, _ in features},
        )
        .sort_values(["task", "obs"])
    )
    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.2), sharey=True)
    for ax, (feature, label) in zip(axes.ravel(), features, strict=True):
        scatter_by_task(
            ax,
            obs_df,
            feature,
            "joint_c2st",
            label,
            ylabel="Mean joint C2ST across seeds",
        )
        ax.set_title(feature, fontsize=8)
    for ax in axes[:, 1:].ravel():
        ax.set_ylabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, -0.01),
    )
    fig.suptitle(
        "Temporary descriptor sweep: seed-averaged observations",
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save_figure(fig, out_dir, "tmp_reference_descriptor_sweep_seed_averaged", formats)


def plot_two_moons_variance_explanation(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    desc_path = Path(
        "pfn_testing/sbi/outputs/layer_ablation/two_moons_reference_descriptors_by_obs.csv"
    )
    if not desc_path.exists():
        print("Skipping two_moons variance explanation: descriptor CSV not found.")
        return

    tm = df[df["task"] == "two_moons"].copy()
    descriptors = pd.read_csv(desc_path)
    features = [
        ("pearson_rank_gap", "|Pearson corr - rank corr|"),
        ("log_det_cov", "Reference log det covariance"),
    ]
    plot_df = tm.merge(
        descriptors[["obs", *(feature for feature, _ in features)]],
        on="obs",
        how="left",
    )
    plot_df["joint_obs_mean"] = plot_df.groupby("obs")["joint_c2st"].transform("mean")
    plot_df["joint_obs_resid"] = plot_df["joint_c2st"] - plot_df["joint_obs_mean"]
    obs_df = (
        plot_df.groupby("obs", as_index=False)
        .agg(
            joint_c2st=("joint_c2st", "mean"),
            **{feature: (feature, "first") for feature, _ in features},
        )
        .sort_values("obs")
    )
    marker_by_seed = seed_markers(plot_df)
    seed_palette = {
        seed: color
        for seed, color in zip(
            marker_by_seed,
            ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442"],
            strict=False,
        )
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25))
    for ax, (feature, xlabel) in zip(axes[:2], features, strict=True):
        for seed, marker in marker_by_seed.items():
            sub = plot_df[plot_df["seed"] == seed]
            ax.scatter(
                sub[feature],
                sub["joint_c2st"],
                s=26,
                marker=marker,
                color=seed_palette[seed],
                alpha=0.42,
                edgecolor="white",
                linewidth=0.3,
                label=f"s{seed}",
            )
        ax.scatter(
            obs_df[feature],
            obs_df["joint_c2st"],
            s=68,
            facecolor="white",
            edgecolor="0.1",
            linewidth=1.0,
            zorder=5,
            label="obs mean",
        )
        for _, row in obs_df.iterrows():
            ax.text(
                row[feature],
                row["joint_c2st"] + 0.005,
                str(int(row["obs"])),
                ha="center",
                va="bottom",
                fontsize=6,
            )
        xs = obs_df[feature].to_numpy(dtype=np.float64)
        ys = obs_df["joint_c2st"].to_numpy(dtype=np.float64)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() >= 3:
            coef = np.polyfit(xs[mask], ys[mask], deg=1)
            grid = np.linspace(xs[mask].min(), xs[mask].max(), 100)
            ax.plot(grid, coef[0] * grid + coef[1], color="0.2", lw=1.1)
            pearson_r = float(np.corrcoef(xs[mask], ys[mask])[0, 1])
            rho, _ = spearmanr(xs[mask], ys[mask])
            ax.text(
                0.04,
                0.96,
                rf"$r$={pearson_r:.2f}, $\rho$={rho:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 2.0},
            )
        ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Joint C2ST")
        ax.set_title("Observation-level reference descriptor")
        ax.grid(alpha=0.18, linewidth=0.6)

    ax = axes[2]
    for seed, marker in marker_by_seed.items():
        sub = plot_df[plot_df["seed"] == seed]
        ax.scatter(
            sub["signed_log_var_error"],
            sub["joint_obs_resid"],
            s=30,
            marker=marker,
            color=seed_palette[seed],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            label=f"s{seed}",
        )
    xs = plot_df["signed_log_var_error"].to_numpy(dtype=np.float64)
    ys = plot_df["joint_obs_resid"].to_numpy(dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() >= 3:
        coef = np.polyfit(xs[mask], ys[mask], deg=1)
        grid = np.linspace(xs[mask].min(), xs[mask].max(), 100)
        ax.plot(grid, coef[0] * grid + coef[1], color="0.2", lw=1.1)
        pearson_r = float(np.corrcoef(xs[mask], ys[mask])[0, 1])
        rho, _ = spearmanr(xs[mask], ys[mask])
        ax.text(
            0.04,
            0.96,
            rf"$r$={pearson_r:.2f}, $\rho$={rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 2.0},
        )
    ax.axhline(0, color="0.45", ls=":", lw=0.8)
    ax.axvline(0, color="0.75", ls=":", lw=0.8)
    ax.set_xlabel("Signed log variance error")
    ax.set_ylabel("Joint C2ST - observation mean")
    ax.set_title("Within-observation seed scatter")
    ax.grid(alpha=0.18, linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, -0.02),
    )
    fig.suptitle("Two moons: what explains continuous joint C2ST variance?", y=1.03)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    save_figure(fig, out_dir, "tmp_two_moons_variance_explanation", formats)


def plot_seed_averaged_structure_location(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    obs_df = aggregate_observations_over_seeds(df)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharey=True)
    errorbar_by_task(
        axes[0],
        obs_df,
        "ref_var_anisotropy_cv_mean",
        "joint_c2st_mean",
        "joint_c2st_sem",
        "Reference posterior anisotropy",
    )
    axes[0].set_title("Intrinsic structure")
    panel_label(axes[0], "A")

    errorbar_by_task(
        axes[1],
        obs_df,
        "mean_rmse_std_mean",
        "joint_c2st_mean",
        "joint_c2st_sem",
        "Mean error / posterior s.d.",
        xerr="mean_rmse_std_sem",
        xscale="log",
    )
    axes[1].set_title("Location error")
    axes[1].set_ylabel("")
    panel_label(axes[1], "B")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, -0.02),
    )
    fig.suptitle(
        "Seed-averaged observations: structure and location error explain C2ST",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    save_figure(fig, out_dir, "tabpfn_seed_averaged_structure_location", formats)


def plot_performance_map(
    df: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2))
    ax = axes[0, 0]
    metrics = ["joint_c2st", "marginal_c2st", "rank_c2st"]
    metric_labels = ["joint", "marginal", "rank"]
    width = 0.22
    xs = np.arange(len(TASK_ORDER))
    for j, metric in enumerate(metrics):
        means = [df.loc[df["task"] == task, metric].mean() for task in TASK_ORDER]
        sem = [
            df.loc[df["task"] == task, metric].std(ddof=1)
            / np.sqrt((df["task"] == task).sum())
            for task in TASK_ORDER
        ]
        ax.bar(
            xs + (j - 1) * width,
            means,
            width=width,
            yerr=sem,
            capsize=2.0,
            color=["#999999", "#BBBBBB", "#666666"][j],
            edgecolor="0.25",
            linewidth=0.5,
            label=metric_labels[j],
        )
    for i, task in enumerate(TASK_ORDER):
        y = df.loc[df["task"] == task, "joint_c2st"].to_numpy()
        jitter = np.linspace(-0.045, 0.045, len(y))
        ax.scatter(
            np.full_like(y, xs[i] - width) + jitter,
            y,
            s=12,
            color=TASK_COLOR[task],
            alpha=0.55,
            edgecolor="none",
        )
    ax.axhline(0.5, color="0.45", lw=0.8, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], rotation=18, ha="right")
    ax.set_ylabel("C2ST")
    ax.set_title("Performance decomposes by task")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    panel_label(ax, "A")

    scatter_specs = [
        (
            axes[0, 1],
            "ref_var_anisotropy_cv",
            "joint_c2st",
            "Reference posterior anisotropy",
            None,
            "Intrinsic structure",
        ),
        (
            axes[0, 2],
            "mean_rmse_std",
            "joint_c2st",
            "Mean error / posterior s.d.",
            "log",
            "Location error",
        ),
        (
            axes[1, 0],
            "abs_log_var_error",
            "joint_c2st",
            "Abs. log marginal variance error",
            None,
            "Scale error",
        ),
        (
            axes[1, 1],
            "cov_rel_fro",
            "joint_c2st",
            "Relative covariance error",
            "log",
            "Covariance error",
        ),
    ]
    for label, (ax, x, y, xlabel, xscale, title) in zip(
        "BCDE", scatter_specs, strict=True
    ):
        scatter_by_task(ax, df, x, y, xlabel, xscale=xscale)
        ax.set_title(title)
        panel_label(ax, label)

    ax = axes[1, 2]
    scatter_by_task(
        ax,
        df,
        "ref_var_anisotropy_cv",
        "joint_gap",
        "Reference posterior anisotropy",
        ylabel="Joint - marginal C2ST",
        fit=True,
    )
    ax.axhline(0, color="0.45", lw=0.8, ls=":")
    ax.set_ylim(-0.02, max(0.28, float(df["joint_gap"].max()) + 0.03))
    ax.set_title("Joint-only failure")
    panel_label(ax, "F")

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, -0.015),
    )
    fig.suptitle(
        "TabPFN-NPE PCA64 performance tracks posterior structure and induced errors",
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(fig, out_dir, "tabpfn_structure_performance_map", formats)


def plot_task_heatmap(
    df: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]
) -> None:
    cols = [
        "joint_c2st",
        "joint_gap",
        "ref_var_anisotropy_cv",
        "ref_log_total_var",
        "mean_rmse_std",
        "quantile_rmse_std",
        "abs_log_var_error",
        "cov_rel_fro",
        "corr_offdiag_mae",
    ]
    labels = [
        "joint\nC2ST",
        "joint-\nmarg",
        "posterior\nanisotropy",
        "log total\nvariance",
        "mean\nerror",
        "quantile\nerror",
        "variance\nerror",
        "covariance\nerror",
        "correlation\nerror",
    ]
    means = df.groupby("task", observed=True)[cols].mean().loc[TASK_ORDER]
    z = (means - means.mean(axis=0)) / means.std(axis=0, ddof=0).replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(9.3, 2.9))
    im = ax.imshow(z.to_numpy(), cmap="PuOr_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(TASK_ORDER)))
    ax.set_yticklabels([TASK_LABEL[t] for t in TASK_ORDER])
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("Task means: structure and error signatures")

    for i, task in enumerate(TASK_ORDER):
        for j, col in enumerate(cols):
            value = means.loc[task, col]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(z.loc[task, col]) > 1.0 else "black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Relative task mean (z-score by column)")
    ax.axvline(1.5, color="0.2", lw=0.8)
    ax.axvline(3.5, color="0.2", lw=0.8)
    ax.text(0.08, -0.35, "performance", transform=ax.transAxes, ha="left", va="top")
    ax.text(
        0.31, -0.35, "intrinsic structure", transform=ax.transAxes, ha="left", va="top"
    )
    ax.text(
        0.62,
        -0.35,
        "model posterior error",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    fig.tight_layout()
    save_figure(fig, out_dir, "tabpfn_task_structure_error_heatmap", formats)


def covariance_ellipse(
    ax: plt.Axes,
    samples: np.ndarray,
    color: str,
    *,
    linestyle: str,
    label: str,
    n_std: float = 2.0,
) -> None:
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ell = Ellipse(
        xy=mean,
        width=2 * n_std * np.sqrt(eigvals[0]),
        height=2 * n_std * np.sqrt(eigvals[1]),
        angle=angle,
        facecolor="none",
        edgecolor=color,
        lw=1.4,
        ls=linestyle,
        label=label,
    )
    ax.add_patch(ell)


def corrcoef_safe(samples: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(np.asarray(samples, dtype=np.float64), rowvar=False)
    corr = np.atleast_2d(corr)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def offdiag(mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] < 2:
        return np.array([], dtype=np.float64)
    return mat[np.triu_indices(mat.shape[0], k=1)]


def reference_samples(task_name: str, obs: int, max_samples: int = 1000) -> np.ndarray:
    key = (task_name, obs)
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = (
            get_task(task_name)
            .get_reference_posterior_samples(num_observation=obs)
            .numpy()
            .astype(np.float64)
        )
    samples = _REFERENCE_CACHE[key]
    if len(samples) > max_samples:
        rng = np.random.default_rng(1234 + obs)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = samples[idx]
    return samples


def project_for_display(
    ref: np.ndarray, model: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if ref.shape[1] == 2:
        return ref, model, r"$\theta_1$", r"$\theta_2$"
    ref_mean = ref.mean(axis=0)
    centered = ref - ref_mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return (
        centered @ basis,
        (model - ref_mean) @ basis,
        "Reference PC1",
        "Reference PC2",
    )


def rank_uniform(samples: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(samples, dtype=np.float64)
    for dim in range(samples.shape[1]):
        ranks[:, dim] = rankdata(samples[:, dim], method="average") / (len(samples) + 1)
    return ranks


def reference_corr_rank_gap(samples: np.ndarray) -> float:
    pearson_corr = corrcoef_safe(samples)
    rank_corr = corrcoef_safe(rank_uniform(samples))
    gap = np.abs(offdiag(pearson_corr) - offdiag(rank_corr))
    if gap.size == 0:
        return 0.0
    return float(np.mean(gap))


def add_reference_corr_rank_gap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        for obs in sorted(df.loc[df["task"] == task, "obs"].unique()):
            ref = reference_samples(str(task), int(obs), max_samples=5000)
            rows.append(
                {
                    "task": str(task),
                    "obs": int(obs),
                    "ref_corr_rank_gap": reference_corr_rank_gap(ref),
                }
            )
    return df.merge(pd.DataFrame(rows), on=["task", "obs"], how="left")


def pooled_rank_transform(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a_out = np.empty_like(a, dtype=np.float64)
    b_out = np.empty_like(b, dtype=np.float64)
    n_a = a.shape[0]
    n_b = b.shape[0]
    for dim in range(a.shape[1]):
        pooled = np.concatenate([a[:, dim], b[:, dim]])
        order = np.argsort(pooled, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(pooled) + 1)
        ranks = ranks / (len(pooled) + 1)
        a_out[:, dim] = ranks[:n_a]
        b_out[:, dim] = ranks[n_a : n_a + n_b]
    return a_out, b_out


def strongest_rank_pair(ref_u: np.ndarray, model_u: np.ndarray) -> tuple[int, int]:
    if ref_u.shape[1] == 2:
        return 0, 1
    ref_corr = corrcoef_safe(ref_u)
    model_corr = corrcoef_safe(model_u)
    diff = np.abs(model_corr - ref_corr)
    np.fill_diagonal(diff, -np.inf)
    i, j = np.unravel_index(np.argmax(diff), diff.shape)
    return int(min(i, j)), int(max(i, j))


def pair_indices(dim: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(dim) for j in range(i + 1, dim)]


def pair_labels(pairs: list[tuple[int, int]]) -> list[str]:
    return [rf"$\theta_{{{i + 1}}}$-$\theta_{{{j + 1}}}$" for i, j in pairs]


def correlation_delta_row(
    ref: np.ndarray, model: np.ndarray
) -> tuple[list[str], np.ndarray]:
    n = min(len(ref), len(model))
    ref_corr = corrcoef_safe(ref[:n])
    model_corr = corrcoef_safe(model[:n])
    pairs = pair_indices(ref_corr.shape[0])
    labels = pair_labels(pairs)
    delta = np.asarray(
        [model_corr[i, j] - ref_corr[i, j] for i, j in pairs], dtype=np.float64
    )
    return labels, delta


def plot_pairwise_correlation_fingerprint(
    df: pd.DataFrame,
    flow_dir: Path,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    task_payloads = []
    flow_cache: dict[tuple[str, int], np.ndarray] = {}

    for task in TASK_ORDER:
        sub = df[df["task"] == task].sort_values(
            ["joint_gap", "rank_c2st", "joint_c2st"],
            ascending=[False, False, False],
        )
        if sub.empty:
            continue

        labels: list[str] | None = None
        deltas = []
        gaps = []
        row_labels = []
        for _, row in sub.iterrows():
            seed = int(row["seed"])
            obs = int(row["obs"])
            cache_key = (task, seed)
            if cache_key not in flow_cache:
                flow_path = flow_dir / f"{task}_s{seed}_{METHOD}.npz"
                flow_cache[cache_key] = np.load(flow_path)["flow_samples"].astype(
                    np.float64
                )

            model = flow_cache[cache_key][obs - 1]
            ref = reference_samples(task, obs, max_samples=len(model))
            row_labels.append(f"s{seed} o{obs}")
            row_pair_labels, delta = correlation_delta_row(ref, model)
            if labels is None:
                labels = row_pair_labels
            deltas.append(delta)
            gaps.append(float(row["joint_gap"]))

        task_payloads.append(
            {
                "task": task,
                "labels": labels or [],
                "deltas": np.vstack(deltas),
                "gaps": np.asarray(gaps, dtype=np.float64),
                "row_labels": row_labels,
            }
        )

    if not task_payloads:
        return

    all_deltas = np.concatenate(
        [payload["deltas"].ravel() for payload in task_payloads]
    )
    delta_vmax = max(0.25, min(1.0, float(np.nanmax(np.abs(all_deltas)))))
    gap_vmax = max(
        0.05, float(np.nanmax([payload["gaps"].max() for payload in task_payloads]))
    )

    fig_height = 1.0 + 2.55 * len(task_payloads)
    fig = plt.figure(figsize=(11.4, fig_height))
    gs = fig.add_gridspec(
        len(task_payloads),
        2,
        width_ratios=[20, 1.1],
        hspace=0.65,
        wspace=0.06,
    )

    heat_axes = []
    gap_axes = []
    delta_im = None
    gap_im = None
    for idx, payload in enumerate(task_payloads):
        ax = fig.add_subplot(gs[idx, 0])
        gap_ax = fig.add_subplot(gs[idx, 1], sharey=ax)
        heat_axes.append(ax)
        gap_axes.append(gap_ax)

        delta_im = ax.imshow(
            payload["deltas"],
            cmap="RdBu_r",
            vmin=-delta_vmax,
            vmax=delta_vmax,
            aspect="auto",
            interpolation="nearest",
        )
        gap_im = gap_ax.imshow(
            payload["gaps"][:, None],
            cmap="viridis",
            vmin=0.0,
            vmax=gap_vmax,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(
            f"{TASK_LABEL[payload['task']]}: pairwise correlation error, sorted by joint-marginal C2ST"
        )
        ax.set_xticks(np.arange(len(payload["labels"])))
        ax.set_xticklabels(payload["labels"], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(payload["row_labels"])))
        ax.set_yticklabels(payload["row_labels"], fontsize=5.5)
        ax.set_ylabel("seed / obs")
        ax.grid(False)

        gap_ax.set_title("gap", fontsize=8)
        gap_ax.set_xticks([])
        gap_ax.tick_params(axis="y", labelleft=False, left=False)
        gap_ax.grid(False)

    if delta_im is not None:
        cbar = fig.colorbar(
            delta_im,
            ax=heat_axes,
            orientation="horizontal",
            fraction=0.04,
            pad=0.08,
            aspect=35,
        )
        cbar.set_label(r"$\Delta\rho_{ij}$ = corr$_{model}$ - corr$_{ref}$")
    if gap_im is not None:
        cbar = fig.colorbar(gap_im, ax=gap_axes, fraction=0.55, pad=0.12)
        cbar.set_label("Joint - marginal C2ST")

    fig.suptitle(
        "Which posterior parameter-pair correlations does TabPFN-NPE miss?", y=0.995
    )
    save_figure(fig, out_dir, "tabpfn_pairwise_correlation_fingerprint", formats)


def plot_geometry_examples(
    df: pd.DataFrame,
    flow_dir: Path,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    rows = []
    for task in TASK_ORDER:
        sub = df[df["task"] == task]
        rows.append(sub.loc[sub["joint_c2st"].idxmax()])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for ax, row in zip(axes, rows, strict=True):
        task = str(row["task"])
        seed = int(row["seed"])
        obs = int(row["obs"])
        flow_path = flow_dir / f"{task}_s{seed}_{METHOD}.npz"
        flow_npz = np.load(flow_path)
        model = flow_npz["flow_samples"][obs - 1].astype(np.float64)
        ref = reference_samples(task, obs, max_samples=len(model))
        ref_2d, model_2d, xlabel, ylabel = project_for_display(ref, model)

        ax.scatter(
            ref_2d[:, 0],
            ref_2d[:, 1],
            s=8,
            color="0.65",
            alpha=0.35,
            edgecolor="none",
            label="reference",
        )
        ax.scatter(
            model_2d[:, 0],
            model_2d[:, 1],
            s=8,
            color=TASK_COLOR[task],
            alpha=0.35,
            edgecolor="none",
            label="TabPFN-NPE",
        )
        covariance_ellipse(ax, ref_2d, "0.2", linestyle="-", label="reference 2 s.d.")
        covariance_ellipse(
            ax, model_2d, TASK_COLOR[task], linestyle="--", label="model 2 s.d."
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{TASK_LABEL[task]}\n"
            f"obs {obs}, seed {seed}: joint={row['joint_c2st']:.2f}, "
            f"gap={row['joint_gap']:.2f}"
        )
        ax.set_aspect("equal", adjustable="datalim")

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Worst-observation posterior geometry: reference vs TabPFN-NPE PCA64", y=1.04
    )
    fig.tight_layout()
    save_figure(fig, out_dir, "tabpfn_posterior_geometry_examples", formats)


def plot_rank_copula_examples(
    df: pd.DataFrame,
    flow_dir: Path,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    rows = []
    for task in TASK_ORDER:
        sub = df[df["task"] == task]
        rows.append(sub.loc[sub["rank_c2st"].idxmax()])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    for ax, row in zip(axes, rows, strict=True):
        task = str(row["task"])
        seed = int(row["seed"])
        obs = int(row["obs"])
        flow_path = flow_dir / f"{task}_s{seed}_{METHOD}.npz"
        flow_npz = np.load(flow_path)
        model = flow_npz["flow_samples"][obs - 1].astype(np.float64)
        ref = reference_samples(task, obs, max_samples=len(model))
        ref_u_all, model_u_all = pooled_rank_transform(ref, model)
        i, j = strongest_rank_pair(ref_u_all, model_u_all)
        ref_u = ref_u_all[:, [i, j]]
        model_u = model_u_all[:, [i, j]]

        ax.scatter(
            ref_u[:, 0],
            ref_u[:, 1],
            s=8,
            color="0.55",
            alpha=0.30,
            edgecolor="none",
            label="reference",
        )
        ax.scatter(
            model_u[:, 0],
            model_u[:, 1],
            s=8,
            color=TASK_COLOR[task],
            alpha=0.32,
            edgecolor="none",
            label="TabPFN-NPE",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(rf"pooled rank($\theta_{{{i + 1}}}$)")
        ax.set_ylabel(rf"pooled rank($\theta_{{{j + 1}}}$)")
        ax.set_title(
            f"{TASK_LABEL[task]}\n"
            f"obs {obs}, seed {seed}: rank={row['rank_c2st']:.2f}, "
            f"joint={row['joint_c2st']:.2f}"
        )
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Pooled-rank view of rank C2ST failures", y=1.04)
    fig.tight_layout()
    save_figure(fig, out_dir, "tabpfn_rank_copula_examples", formats)


def plot_within_task_residuals(
    df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    specs = [
        ("mean_rmse_std", "Mean error / posterior s.d.", "log"),
        ("quantile_rmse_std", "Quantile RMSE / posterior s.d.", "log"),
        ("abs_log_var_error", "Abs. log variance error", None),
        ("cov_rel_fro", "Relative covariance error", "log"),
    ]
    residual_df = df.copy()
    for col in ["joint_c2st", *(spec[0] for spec in specs)]:
        residual_df[f"{col}_resid"] = residual_df[col] - residual_df.groupby("task")[
            col
        ].transform("mean")

    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9), sharey=True)
    for ax, (metric, xlabel, xscale) in zip(axes, specs, strict=True):
        x_col = f"{metric}_resid"
        y_col = "joint_c2st_resid"
        for task in TASK_ORDER:
            sub = residual_df[residual_df["task"] == task]
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=24,
                color=TASK_COLOR[task],
                edgecolor="white",
                linewidth=0.3,
                alpha=0.84,
                label=TASK_LABEL[task],
            )
        xs = residual_df[x_col].to_numpy(float)
        ys = residual_df[y_col].to_numpy(float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() >= 3:
            coef = np.polyfit(xs[mask], ys[mask], deg=1)
            grid = np.linspace(xs[mask].min(), xs[mask].max(), 100)
            ax.plot(grid, coef[0] * grid + coef[1], color="0.25", lw=1.1)
            rho, _ = spearmanr(xs[mask], ys[mask])
            ax.text(
                0.04,
                0.96,
                rf"$\rho$={rho:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 2.0},
            )
        if xscale == "log":
            ax.set_xlabel(f"Residual {xlabel}")
        else:
            ax.set_xlabel(f"Residual {xlabel}")
        ax.axhline(0, color="0.5", lw=0.8, ls=":")
        ax.axvline(0, color="0.5", lw=0.8, ls=":")
    axes[0].set_ylabel("Residual joint C2ST")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, -0.06),
    )
    fig.suptitle(
        "Within-task residuals: errors still track C2ST after removing task means",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure(fig, out_dir, "tabpfn_within_task_error_residuals", formats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-csv", type=Path, default=DEFAULT_PROBE)
    ap.add_argument("--flow-dir", type=Path, default=DEFAULT_FLOW_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Output formats for each figure.",
    )
    args = ap.parse_args()

    configure_style()
    df = load_probe(args.probe_csv)
    formats = tuple(args.formats)
    plot_performance_map(df, args.out_dir, formats)
    plot_seed_level_structure_location(df, args.out_dir, formats)
    plot_reference_geometry_score(df, args.out_dir, formats)
    plot_reference_anisotropy(df, args.out_dir, formats)
    plot_reference_anisotropy_seed_averaged(df, args.out_dir, formats)
    plot_reference_marginal_variance_sd(df, args.out_dir, formats)
    plot_reference_marginal_variance_sd_seed_averaged(df, args.out_dir, formats)
    plot_reference_corr_rank_gap(df, args.out_dir, formats)
    plot_reference_descriptor_sweep(df, args.out_dir, formats)
    plot_two_moons_variance_explanation(df, args.out_dir, formats)
    plot_task_heatmap(df, args.out_dir, formats)
    plot_geometry_examples(df, args.flow_dir, args.out_dir, formats)
    plot_rank_copula_examples(df, args.flow_dir, args.out_dir, formats)
    plot_pairwise_correlation_fingerprint(df, args.flow_dir, args.out_dir, formats)
    plot_within_task_residuals(df, args.out_dir, formats)
    print(f"Wrote figures to {args.out_dir}")
    for stem in [
        "tabpfn_structure_performance_map",
        "tabpfn_seed_level_structure_location",
        "tabpfn_reference_geometry_score",
        "tabpfn_reference_anisotropy",
        "tabpfn_reference_anisotropy_seed_averaged",
        "tabpfn_reference_marginal_variance_sd",
        "tabpfn_reference_marginal_variance_sd_seed_averaged",
        "tabpfn_reference_corr_rank_gap",
        "tmp_reference_descriptor_sweep_seed_level",
        "tmp_reference_descriptor_sweep_seed_averaged",
        "tmp_two_moons_variance_explanation",
        "tabpfn_task_structure_error_heatmap",
        "tabpfn_posterior_geometry_examples",
        "tabpfn_rank_copula_examples",
        "tabpfn_pairwise_correlation_fingerprint",
        "tabpfn_within_task_error_residuals",
    ]:
        print(" ", stem)


if __name__ == "__main__":
    main()
