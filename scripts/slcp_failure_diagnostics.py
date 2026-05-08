"""Prototype PFN-NPE failure diagnostics for the SLCP case study.

Reads saved posterior samples from:
  pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile/{task}_s{seed}_{method}*.npz

Writes:
  pfn_testing/sbi/outputs/layer_ablation/failure_diagnostics/

The diagnostics are intentionally post-processing only:
  - budget-overlaid marginal KDEs for one reference observation
  - 2D KDE contours for the two SLCP location parameters
  - moment error trajectories across budgets
  - whitened posterior diagnostics
  - reference-observation SBC-style rank histograms
  - HDR 50/80 volume-proxy ratios

The rank histograms use the fixed sbibm reference observations and saved
theta_true values, so they are a fixed-reference calibration diagnostic rather
than full SBC over newly simulated observations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task  # noqa: E402

SAMPLE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/failure_diagnostics")

PARAM_LABELS = {
    "slcp": [
        r"$\theta_0$: mean 1",
        r"$\theta_1$: mean 2",
        r"$\theta_2$: scale 1",
        r"$\theta_3$: scale 2",
        r"$\theta_4$: corr.",
    ],
    "ar1_ts_t50": [
        r"$\rho$",
        r"$\log\sigma$",
    ],
}
TASK_LABELS = {
    "slcp": "SLCP",
    "ar1_ts_t50": "AR(1)",
}
TASK_2D_TITLES = {
    "slcp": "2D posterior shape for location parameters",
    "ar1_ts_t50": r"2D posterior shape for $(\rho, \log\sigma)$",
}
METHOD_LABELS = {
    "nsf": "PFN-NPE",
    "npe_pfn": "NPE-PFN",
}
METHOD_COLORS = {
    "nsf": "#0072B2",
    "npe_pfn": "#D55E00",
}


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task)


def param_label(task: str, dim: int) -> str:
    labels = PARAM_LABELS.get(task, [])
    return labels[dim] if dim < len(labels) else f"dim {dim}"

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


@dataclass(frozen=True)
class Run:
    task: str
    method: str
    seed: int
    budget: int
    path: Path
    flow_samples: np.ndarray
    theta_true: np.ndarray
    x_ref: np.ndarray


def budget_from_path(path: Path, task: str, seed: int, method: str) -> int | None:
    prefix = f"{task}_s{seed}_{method}"
    suffix = path.stem.removeprefix(prefix)
    if path.stem == prefix:
        return 10_000
    if suffix.startswith("_n") and suffix[2:].isdigit():
        return int(suffix[2:])
    return None


def discover_runs(
    task: str,
    method: str,
    seeds: list[int],
    budgets: list[int] | None,
    sample_dir: Path,
) -> list[Run]:
    runs: list[Run] = []
    wanted_budgets = set(budgets) if budgets is not None else None
    for seed in seeds:
        prefix = f"{task}_s{seed}_{method}"
        for path in sorted(sample_dir.glob(f"{prefix}*.npz")):
            budget = budget_from_path(path, task, seed, method)
            if budget is None:
                continue
            if wanted_budgets is not None and budget not in wanted_budgets:
                continue
            loaded = np.load(path, allow_pickle=True)
            if "flow_samples" not in loaded:
                continue
            runs.append(
                Run(
                    task=task,
                    method=method,
                    seed=seed,
                    budget=budget,
                    path=path,
                    flow_samples=np.asarray(loaded["flow_samples"], dtype=float),
                    theta_true=np.asarray(loaded["theta_true"], dtype=float),
                    x_ref=np.asarray(loaded["x_ref"], dtype=float),
                )
            )
    if not runs:
        raise FileNotFoundError(
            f"No runs found in {sample_dir} for task={task}, method={method}, "
            f"seeds={seeds}, budgets={budgets or 'auto'}"
        )
    return sorted(runs, key=lambda r: (r.budget, r.seed))


def load_reference_samples(task_name: str, n_ref: int) -> list[np.ndarray]:
    task = get_task(task_name)
    return [
        task.get_reference_posterior_samples(num_observation=i + 1).numpy()
        for i in range(n_ref)
    ]


def kde_curve(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    if len(samples) < 3 or np.std(samples) <= 1e-12:
        y = np.zeros_like(grid)
        y[np.argmin(np.abs(grid - np.mean(samples)))] = 1.0
        return y
    return gaussian_kde(samples)(grid)


def kde2d_on_grid(
    samples: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    xx, yy = np.meshgrid(x_grid, y_grid)
    points = np.vstack([xx.ravel(), yy.ravel()])
    if len(samples) < 3:
        return np.zeros_like(xx)
    try:
        density = gaussian_kde(samples.T)(points).reshape(xx.shape)
    except np.linalg.LinAlgError:
        jitter = np.random.default_rng(0).normal(0.0, 1e-6, size=samples.shape)
        density = gaussian_kde((samples + jitter).T)(points).reshape(xx.shape)
    return density


def density_threshold_for_mass(density: np.ndarray, mass: float) -> float:
    flat = np.sort(np.ravel(density))[::-1]
    total = float(flat.sum())
    if total <= 0.0:
        return float("nan")
    cdf = np.cumsum(flat) / total
    idx = min(int(np.searchsorted(cdf, mass, side="left")), len(flat) - 1)
    return float(flat[idx])


def covariance(samples: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    cov = np.cov(samples, rowvar=False)
    scale = float(np.trace(cov) / max(cov.shape[0], 1))
    jitter = eps * max(scale, 1.0)
    return cov + jitter * np.eye(cov.shape[0])


def whiten_against_reference(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu = ref.mean(axis=0)
    cov_ref = covariance(ref)
    vals, vecs = np.linalg.eigh(cov_ref)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-10))) @ vecs.T
    return (pred - mu) @ inv_sqrt


def knn_density_proxy(samples: np.ndarray, k: int = 20) -> np.ndarray:
    """Return a fast local-density proxy; larger means denser."""
    center = np.median(samples, axis=0)
    scale = np.quantile(samples, 0.75, axis=0) - np.quantile(samples, 0.25, axis=0)
    scale = np.where(scale > 1e-12, scale, samples.std(axis=0, ddof=1) + 1e-12)
    z = (samples - center) / scale
    n = len(z)
    k_eff = min(max(2, k), n)
    dists, _ = cKDTree(z).query(z, k=k_eff)
    kth_dist = dists[:, -1] if dists.ndim == 2 else dists
    return -kth_dist


def hdr_log_volume_proxy(samples: np.ndarray, mass: float) -> float:
    """Approximate HDR log-volume with kNN top-mass covariance volume.

    Select the highest-density `mass` fraction of samples under a kNN proxy,
    then use 0.5 * logdet(covariance(selected)) as a region-size proxy. The
    omitted constants cancel in ratios between PFN samples and references.
    """
    n = len(samples)
    k = max(samples.shape[1] + 2, int(np.ceil(mass * n)))
    density = knn_density_proxy(samples)
    idx = np.argsort(density)[-k:]
    cov = covariance(samples[idx])
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("nan")
    return 0.5 * float(logdet)


def compute_metrics(
    runs: list[Run],
    refs: list[np.ndarray],
    masses: list[float],
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    moment_rows: list[dict[str, float | int | str]] = []
    global_rows: list[dict[str, float | int | str]] = []
    for run in runs:
        n_ref = min(len(refs), run.flow_samples.shape[0])
        for obs_idx in range(n_ref):
            pred = run.flow_samples[obs_idx]
            ref = refs[obs_idx]
            dim_theta = pred.shape[1]

            ref_mean = ref.mean(axis=0)
            ref_sd = ref.std(axis=0, ddof=1) + 1e-12
            pred_mean = pred.mean(axis=0)
            pred_sd = pred.std(axis=0, ddof=1) + 1e-12
            ref_iqr = np.quantile(ref, 0.75, axis=0) - np.quantile(ref, 0.25, axis=0)
            pred_iqr = np.quantile(pred, 0.75, axis=0) - np.quantile(pred, 0.25, axis=0)
            ref_w90 = np.quantile(ref, 0.95, axis=0) - np.quantile(ref, 0.05, axis=0)
            pred_w90 = np.quantile(pred, 0.95, axis=0) - np.quantile(pred, 0.05, axis=0)

            z = whiten_against_reference(pred, ref)
            z_cov = covariance(z)
            diag_err = np.diag(z_cov) - 1.0
            offdiag = z_cov - np.diag(np.diag(z_cov))
            eigvals = np.linalg.eigvalsh(z_cov)
            global_row: dict[str, float | int | str] = {
                "task": run.task,
                "method": run.method,
                "seed": run.seed,
                "budget": run.budget,
                "obs": obs_idx + 1,
                "whitened_mean_norm": float(np.linalg.norm(z.mean(axis=0))),
                "whitened_diag_rmse": float(np.sqrt(np.mean(diag_err**2))),
                "whitened_offdiag_rmse": float(np.sqrt(np.mean(offdiag**2))),
                "whitened_eig_min": float(np.min(eigvals)),
                "whitened_eig_max": float(np.max(eigvals)),
            }
            for mass in masses:
                ref_log_vol = hdr_log_volume_proxy(ref, mass)
                pred_log_vol = hdr_log_volume_proxy(pred, mass)
                global_row[f"hdr{int(100 * mass)}_log_ratio"] = (
                    float(pred_log_vol - ref_log_vol)
                    if np.isfinite(ref_log_vol) and np.isfinite(pred_log_vol)
                    else float("nan")
                )
            global_rows.append(global_row)

            for dim in range(dim_theta):
                rank = np.sum(pred[:, dim] < run.theta_true[obs_idx, dim])
                moment_rows.append(
                    {
                        "task": run.task,
                        "method": run.method,
                        "seed": run.seed,
                        "budget": run.budget,
                        "obs": obs_idx + 1,
                        "dim": dim,
                        "mean_error_z": float(
                            (pred_mean[dim] - ref_mean[dim]) / ref_sd[dim]
                        ),
                        "abs_mean_error_z": float(
                            abs((pred_mean[dim] - ref_mean[dim]) / ref_sd[dim])
                        ),
                        "log_sd_ratio": float(np.log(pred_sd[dim] / ref_sd[dim])),
                        "log_iqr_ratio": float(
                            np.log((pred_iqr[dim] + 1e-12) / (ref_iqr[dim] + 1e-12))
                        ),
                        "log_w90_ratio": float(
                            np.log((pred_w90[dim] + 1e-12) / (ref_w90[dim] + 1e-12))
                        ),
                        "rank_pit": float((rank + 0.5) / (len(pred) + 1.0)),
                    }
                )
    return moment_rows, global_rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def mean_by_budget_dim(
    rows: list[dict[str, float | int | str]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    budgets = np.array(sorted({int(r["budget"]) for r in rows}), dtype=int)
    dims = np.array(sorted({int(r["dim"]) for r in rows}), dtype=int)
    arr = np.full((len(budgets), len(dims)), np.nan)
    for i, budget in enumerate(budgets):
        for j, dim in enumerate(dims):
            vals = [
                float(r[metric])
                for r in rows
                if int(r["budget"]) == budget and int(r["dim"]) == dim
            ]
            if vals:
                arr[i, j] = float(np.nanmean(vals))
    return budgets, arr


def mean_sem_by_budget(
    rows: list[dict[str, float | int | str]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    budgets = np.array(sorted({int(r["budget"]) for r in rows}), dtype=int)
    means = np.full(len(budgets), np.nan)
    sems = np.full(len(budgets), np.nan)
    for i, budget in enumerate(budgets):
        vals = np.array(
            [float(r[metric]) for r in rows if int(r["budget"]) == budget],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if len(vals):
            means[i] = float(vals.mean())
            sems[i] = (
                float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            )
    return budgets, means, sems


def plot_kde_budget_overlay(
    runs: list[Run],
    refs: list[np.ndarray],
    out_dir: Path,
    obs: int,
) -> None:
    obs_idx = obs - 1
    budgets = sorted({r.budget for r in runs})
    dim_theta = runs[0].flow_samples.shape[2]
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(budgets)))

    fig, axes = plt.subplots(dim_theta, 1, figsize=(7.4, 1.7 * dim_theta), sharex=False)
    if dim_theta == 1:
        axes = [axes]
    ref = refs[obs_idx]
    for dim, ax in enumerate(axes):
        pooled_by_budget = {
            budget: np.concatenate(
                [r.flow_samples[obs_idx, :, dim] for r in runs if r.budget == budget]
            )
            for budget in budgets
        }
        all_values = [ref[:, dim], *pooled_by_budget.values()]
        lo = min(float(np.quantile(v, 0.002)) for v in all_values)
        hi = max(float(np.quantile(v, 0.998)) for v in all_values)
        pad = 0.08 * max(hi - lo, 1e-6)
        grid = np.linspace(lo - pad, hi + pad, 400)

        ax.plot(
            grid, kde_curve(ref[:, dim], grid), color="black", lw=2.0, label="reference"
        )
        ax.axvline(ref[:, dim].mean(), color="black", lw=1.1, ls="--", alpha=0.75)
        for color, budget in zip(colors, budgets, strict=True):
            samples = pooled_by_budget[budget]
            ax.plot(
                grid,
                kde_curve(samples, grid),
                color=color,
                lw=1.7,
                label=f"n={budget:g}",
            )
            ax.axvline(samples.mean(), color=color, lw=0.8, alpha=0.65)
        ax.set_ylabel(param_label(runs[0].task, dim))
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(ncol=3, frameon=False, loc="upper right")
    axes[-1].set_xlabel("Parameter value")
    fig.suptitle(
        f"{task_label(runs[0].task)} {runs[0].method}: marginal KDEs for reference obs {obs}"
    )
    fig.tight_layout()
    out = out_dir / f"{runs[0].task}_{runs[0].method}_obs{obs}_kde_budget_overlay.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_kde_method_panel(
    method_runs: dict[str, list[Run]],
    refs: list[np.ndarray],
    out_dir: Path,
    obs: int,
) -> None:
    obs_idx = obs - 1
    methods = list(method_runs)
    if len(methods) < 2:
        return
    shared_budgets = sorted(
        set.intersection(
            *[set(r.budget for r in runs) for runs in method_runs.values()]
        )
    )
    if not shared_budgets:
        raise ValueError("Cannot make KDE panel: no shared budgets across methods.")

    first_run = next(iter(method_runs.values()))[0]
    dim_theta = first_run.flow_samples.shape[2]
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(shared_budgets)))
    fig, axes = plt.subplots(
        dim_theta,
        len(methods),
        figsize=(4.9 * len(methods), 1.7 * dim_theta + 0.65),
        sharex=False,
        sharey="row",
        squeeze=False,
    )
    ref = refs[obs_idx]
    legend_handles: list[Line2D] = []

    for col, method in enumerate(methods):
        runs = method_runs[method]
        for dim in range(dim_theta):
            ax = axes[dim][col]
            pooled_by_budget = {
                budget: np.concatenate(
                    [
                        r.flow_samples[obs_idx, :, dim]
                        for r in runs
                        if r.budget == budget
                    ]
                )
                for budget in shared_budgets
            }
            all_values = [ref[:, dim], *pooled_by_budget.values()]
            lo = min(float(np.quantile(v, 0.002)) for v in all_values)
            hi = max(float(np.quantile(v, 0.998)) for v in all_values)
            pad = 0.08 * max(hi - lo, 1e-6)
            grid = np.linspace(lo - pad, hi + pad, 400)

            ax.plot(grid, kde_curve(ref[:, dim], grid), color="black", lw=1.8)
            ax.axvline(ref[:, dim].mean(), color="black", lw=0.9, ls="--", alpha=0.75)
            for color, budget in zip(colors, shared_budgets, strict=True):
                samples = pooled_by_budget[budget]
                ax.plot(grid, kde_curve(samples, grid), color=color, lw=1.5)
                ax.axvline(samples.mean(), color=color, lw=0.7, alpha=0.55)
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(
                    param_label(first_run.task, dim)
                )
            if dim == 0:
                ax.set_title(METHOD_LABELS.get(method, method))
            if dim == dim_theta - 1:
                ax.set_xlabel("Parameter value")

    legend_handles.append(Line2D([], [], color="black", lw=1.8, label="reference"))
    for color, budget in zip(colors, shared_budgets, strict=True):
        legend_handles.append(
            Line2D([], [], color=color, lw=1.6, label=f"n={budget:g}")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(f"{task_label(first_run.task)}: marginal KDEs for reference obs {obs}")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    method_slug = "_vs_".join(methods)
    out = out_dir / f"{first_run.task}_{method_slug}_obs{obs}_kde_budget_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_location_2d_kde_panel(
    method_runs: dict[str, list[Run]],
    refs: list[np.ndarray],
    out_dir: Path,
    obs: int,
    dims: tuple[int, int] = (0, 1),
) -> None:
    obs_idx = obs - 1
    methods = list(method_runs)
    if len(methods) < 2:
        return
    shared_budgets = sorted(
        set.intersection(
            *[set(r.budget for r in runs) for runs in method_runs.values()]
        )
    )
    if not shared_budgets:
        raise ValueError("Cannot make 2D KDE panel: no shared budgets across methods.")

    first_run = next(iter(method_runs.values()))[0]
    ref = refs[obs_idx][:, list(dims)]
    all_samples = [ref]
    for runs in method_runs.values():
        for run in runs:
            if run.budget in shared_budgets:
                all_samples.append(run.flow_samples[obs_idx][:, list(dims)])
    pooled = np.vstack(all_samples)
    x_lo, y_lo = np.quantile(pooled, 0.002, axis=0)
    x_hi, y_hi = np.quantile(pooled, 0.998, axis=0)
    x_pad = 0.08 * max(float(x_hi - x_lo), 1e-6)
    y_pad = 0.08 * max(float(y_hi - y_lo), 1e-6)
    x_grid = np.linspace(float(x_lo - x_pad), float(x_hi + x_pad), 170)
    y_grid = np.linspace(float(y_lo - y_pad), float(y_hi + y_pad), 170)

    ref_density = kde2d_on_grid(ref, x_grid, y_grid)
    ref_levels = [
        density_threshold_for_mass(ref_density, 0.80),
        density_threshold_for_mass(ref_density, 0.50),
    ]
    ref_mean = ref.mean(axis=0)

    fig, axes = plt.subplots(
        len(shared_budgets),
        len(methods),
        figsize=(4.25 * len(methods), 3.0 * len(shared_budgets)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, budget in enumerate(shared_budgets):
        for col, method in enumerate(methods):
            ax = axes[row][col]
            runs = [run for run in method_runs[method] if run.budget == budget]
            samples = np.vstack(
                [run.flow_samples[obs_idx][:, list(dims)] for run in runs]
            )
            density = kde2d_on_grid(samples, x_grid, y_grid)
            levels = [
                density_threshold_for_mass(density, 0.80),
                density_threshold_for_mass(density, 0.50),
            ]
            max_density = float(np.nanmax(density))
            color = METHOD_COLORS.get(method, f"C{col}")
            if np.isfinite(levels[0]) and levels[0] < max_density:
                ax.contourf(
                    x_grid,
                    y_grid,
                    density,
                    levels=[levels[0], max_density],
                    colors=[color],
                    alpha=0.12,
                    antialiased=True,
                )
            if np.isfinite(levels[1]) and levels[1] < max_density:
                ax.contourf(
                    x_grid,
                    y_grid,
                    density,
                    levels=[levels[1], max_density],
                    colors=[color],
                    alpha=0.18,
                    antialiased=True,
                )
            if all(np.isfinite(levels)) and levels[0] < levels[1] < max_density:
                ax.contour(
                    x_grid,
                    y_grid,
                    density,
                    levels=levels,
                    colors=[color],
                    linewidths=[1.0, 1.5],
                )
            if all(np.isfinite(ref_levels)):
                ax.contour(
                    x_grid,
                    y_grid,
                    ref_density,
                    levels=ref_levels,
                    colors=["black"],
                    linewidths=[1.0, 1.4],
                    linestyles=["--", "-"],
                )
            sample_mean = samples.mean(axis=0)
            ax.scatter(
                ref_mean[0],
                ref_mean[1],
                marker="o",
                s=20,
                color="black",
                zorder=5,
            )
            ax.scatter(
                sample_mean[0],
                sample_mean[1],
                marker="x",
                s=28,
                color=color,
                linewidths=1.4,
                zorder=6,
            )
            ax.grid(True, alpha=0.18)
            ax.set_aspect("equal", adjustable="box")
            if row == 0:
                ax.set_title(METHOD_LABELS.get(method, method))
            if col == 0:
                ax.set_ylabel(f"n={budget:g}\n{param_label(first_run.task, dims[1])}")
            if row == len(shared_budgets) - 1:
                ax.set_xlabel(param_label(first_run.task, dims[0]))

    handles = [
        Line2D([], [], color="black", lw=1.4, label="reference 50/80% KDE contours"),
        Line2D([], [], marker="o", color="black", lw=0, label="reference mean"),
    ]
    for method in methods:
        color = METHOD_COLORS.get(method, "C0")
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                lw=1.5,
                label=f"{METHOD_LABELS.get(method, method)} 50/80% KDE contours",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"{task_label(first_run.task)}: {TASK_2D_TITLES.get(first_run.task, '2D posterior shape')}, obs {obs}"
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    method_slug = "_vs_".join(methods)
    out = out_dir / f"{first_run.task}_{method_slug}_obs{obs}_location2d_kde_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_moment_trajectories(
    rows: list[dict[str, float | int | str]],
    task: str,
    method: str,
    out_dir: Path,
) -> None:
    metrics = [
        ("mean_error_z", "Mean error / ref SD"),
        ("abs_mean_error_z", "Absolute mean error / ref SD"),
        ("log_sd_ratio", "log(SD ratio)"),
        ("log_iqr_ratio", "log(IQR ratio)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        budgets, arr = mean_by_budget_dim(rows, metric)
        for dim in range(arr.shape[1]):
            label = param_label(task, dim)
            ax.plot(budgets, arr[:, dim], marker="o", lw=1.6, label=label)
        ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    for ax in axes[-2:]:
        ax.set_xlabel("Simulation budget")
    axes[0].legend(ncol=1, frameon=False, fontsize=7)
    fig.suptitle(f"{task_label(task)} {method}: moment error trajectories")
    fig.tight_layout()
    out = out_dir / f"{task}_{method}_moment_trajectories.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_moment_method_panel(
    method_rows: dict[str, list[dict[str, float | int | str]]],
    task: str,
    out_dir: Path,
) -> None:
    methods = list(method_rows)
    if len(methods) < 2:
        return
    shared_budgets = sorted(
        set.intersection(
            *[set(int(r["budget"]) for r in rows) for rows in method_rows.values()]
        )
    )
    metrics = [
        ("mean_error_z", "Mean error / ref SD"),
        ("abs_mean_error_z", "Absolute mean error / ref SD"),
        ("log_sd_ratio", "log(SD ratio)"),
        ("log_iqr_ratio", "log(IQR ratio)"),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        len(methods),
        figsize=(4.9 * len(methods), 2.35 * len(metrics) + 0.35),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for col, method in enumerate(methods):
        rows = [
            row for row in method_rows[method] if int(row["budget"]) in shared_budgets
        ]
        for row_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx][col]
            budgets, arr = mean_by_budget_dim(rows, metric)
            for dim in range(arr.shape[1]):
                label = param_label(task, dim)
                ax.plot(budgets, arr[:, dim], marker="o", lw=1.35, label=label)
            ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(title)
            if row_idx == 0:
                ax.set_title(METHOD_LABELS.get(method, method))
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Simulation budget")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=8,
    )
    fig.suptitle(f"{task_label(task)}: marginal moment diagnostics")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    method_slug = "_vs_".join(methods)
    out = out_dir / f"{task}_{method_slug}_moment_trajectories_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_moment_hdr_method_panel(
    method_moment_rows: dict[str, list[dict[str, float | int | str]]],
    method_global_rows: dict[str, list[dict[str, float | int | str]]],
    task: str,
    masses: list[float],
    out_dir: Path,
) -> None:
    methods = list(method_moment_rows)
    if len(methods) < 2:
        return
    shared_budgets = sorted(
        set.intersection(
            *[
                set(int(r["budget"]) for r in method_moment_rows[method])
                & set(int(r["budget"]) for r in method_global_rows[method])
                for method in methods
            ]
        )
    )
    metrics = [
        ("mean_error_z", "Mean error / ref SD"),
        ("abs_mean_error_z", "Absolute mean error / ref SD"),
        ("log_sd_ratio", "log(SD ratio)"),
        ("log_iqr_ratio", "log(IQR ratio)"),
    ]
    n_rows = len(metrics) + 1
    fig, axes = plt.subplots(
        n_rows,
        len(methods),
        figsize=(4.9 * len(methods), 2.05 * n_rows + 0.65),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for col, method in enumerate(methods):
        moment_rows = [
            row
            for row in method_moment_rows[method]
            if int(row["budget"]) in shared_budgets
        ]
        global_rows = [
            row
            for row in method_global_rows[method]
            if int(row["budget"]) in shared_budgets
        ]
        for row_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx][col]
            budgets, arr = mean_by_budget_dim(moment_rows, metric)
            for dim in range(arr.shape[1]):
                ax.plot(
                    budgets,
                    arr[:, dim],
                    marker="o",
                    lw=1.3,
                    label=param_label(task, dim),
                )
            ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(title)
            if row_idx == 0:
                ax.set_title(METHOD_LABELS.get(method, method))

        ax = axes[-1][col]
        for mass, color in zip(masses, ["C0", "C3", "C2", "C4"], strict=False):
            metric = f"hdr{int(100 * mass)}_log_ratio"
            budgets, means, sems = mean_sem_by_budget(global_rows, metric)
            ax.plot(
                budgets,
                means,
                marker="o",
                lw=1.45,
                color=color,
                label=f"HDR {int(100 * mass)}",
            )
            ax.fill_between(
                budgets,
                means - sems,
                means + sems,
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Simulation budget")
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("log(HDR volume\nproxy / ref.)")

    handles: list[Line2D] = []
    labels: list[str] = []
    for ax in (axes[0][0], axes[-1][0]):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 7),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=8,
    )
    fig.suptitle(f"{task_label(task)}: marginal moments and HDR region size")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    method_slug = "_vs_".join(methods)
    out = out_dir / f"{task}_{method_slug}_moment_hdr_trajectories_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_whitened_diagnostics(
    rows: list[dict[str, float | int | str]],
    task: str,
    method: str,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    axes = axes.ravel()
    metrics = [
        ("whitened_mean_norm", "Whitened mean norm"),
        ("whitened_diag_rmse", "Cov diagonal RMSE from 1"),
        ("whitened_offdiag_rmse", "Cov off-diagonal RMSE"),
    ]
    for ax, (metric, title) in zip(axes[:3], metrics, strict=True):
        budgets, means, sems = mean_sem_by_budget(rows, metric)
        ax.plot(budgets, means, marker="o", color="C0", lw=1.8)
        ax.fill_between(
            budgets, means - sems, means + sems, color="C0", alpha=0.18, linewidth=0
        )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    ax = axes[3]
    budgets, lo, lo_sem = mean_sem_by_budget(rows, "whitened_eig_min")
    _, hi, hi_sem = mean_sem_by_budget(rows, "whitened_eig_max")
    ax.plot(budgets, lo, marker="o", color="C2", lw=1.8, label="min eigenvalue")
    ax.fill_between(
        budgets, lo - lo_sem, lo + lo_sem, color="C2", alpha=0.16, linewidth=0
    )
    ax.plot(budgets, hi, marker="o", color="C3", lw=1.8, label="max eigenvalue")
    ax.fill_between(
        budgets, hi - hi_sem, hi + hi_sem, color="C3", alpha=0.16, linewidth=0
    )
    ax.axhline(1.0, color="0.35", lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_title("Whitened covariance eigenvalue range")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    for ax in axes[-2:]:
        ax.set_xlabel("Simulation budget")
    fig.suptitle(f"{task_label(task)} {method}: whitened posterior diagnostics")
    fig.tight_layout()
    out = out_dir / f"{task}_{method}_whitened_diagnostics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_hdr_ratios(
    rows: list[dict[str, float | int | str]],
    task: str,
    method: str,
    masses: list[float],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for mass, color in zip(masses, ["C0", "C3", "C2", "C4"], strict=False):
        metric = f"hdr{int(100 * mass)}_log_ratio"
        budgets, means, sems = mean_sem_by_budget(rows, metric)
        ax.plot(
            budgets,
            means,
            marker="o",
            lw=1.9,
            color=color,
            label=f"HDR {int(100 * mass)}",
        )
        ax.fill_between(
            budgets, means - sems, means + sems, color=color, alpha=0.16, linewidth=0
        )
    ax.axhline(0.0, color="0.35", lw=0.9, ls=":", label="matched region size")
    ax.set_xscale("log")
    ax.set_xlabel("Simulation budget")
    ax.set_ylabel("log(PFN HDR volume proxy / reference)")
    ax.set_title(f"{task_label(task)} {method}: HDR region-size ratios")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = out_dir / f"{task}_{method}_hdr_ratios.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_hdr_method_panel(
    method_rows: dict[str, list[dict[str, float | int | str]]],
    task: str,
    masses: list[float],
    out_dir: Path,
) -> None:
    methods = list(method_rows)
    if len(methods) < 2:
        return
    shared_budgets = sorted(
        set.intersection(
            *[set(int(r["budget"]) for r in rows) for rows in method_rows.values()]
        )
    )
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(4.9 * len(methods), 4.05),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = ["C0", "C3", "C2", "C4"]
    for col, method in enumerate(methods):
        ax = axes[0][col]
        rows = [
            row for row in method_rows[method] if int(row["budget"]) in shared_budgets
        ]
        for mass, color in zip(masses, colors, strict=False):
            metric = f"hdr{int(100 * mass)}_log_ratio"
            budgets, means, sems = mean_sem_by_budget(rows, metric)
            ax.plot(
                budgets,
                means,
                marker="o",
                lw=1.8,
                color=color,
                label=f"HDR {int(100 * mass)}",
            )
            ax.fill_between(
                budgets,
                means - sems,
                means + sems,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
        ax.set_xscale("log")
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("Simulation budget")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("log(HDR volume proxy / reference)")
    axes[0][0].legend(frameon=False)
    fig.suptitle(f"{task_label(task)}: HDR region-size ratios")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    method_slug = "_vs_".join(methods)
    out = out_dir / f"{task}_{method_slug}_hdr_ratios_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_rank_histograms(
    rows: list[dict[str, float | int | str]],
    task: str,
    method: str,
    out_dir: Path,
) -> None:
    budgets = sorted({int(r["budget"]) for r in rows})
    dims = sorted({int(r["dim"]) for r in rows})
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(budgets)))
    n_cols = min(3, len(dims))
    n_rows = int(np.ceil(len(dims) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), squeeze=False
    )
    bins = np.linspace(0.0, 1.0, 11)
    for idx, dim in enumerate(dims):
        ax = axes[idx // n_cols][idx % n_cols]
        for color, budget in zip(colors, budgets, strict=True):
            vals = [
                float(r["rank_pit"])
                for r in rows
                if int(r["budget"]) == budget and int(r["dim"]) == dim
            ]
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.4,
                color=color,
                label=f"n={budget:g}",
            )
        ax.axhline(1.0, color="0.25", ls=":", lw=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(param_label(task, dim))
        ax.set_xlabel("Rank / posterior sample count")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.2)
    for idx in range(len(dims), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{task_label(task)} {method}: reference-observation SBC-style ranks")
    fig.tight_layout()
    out = out_dir / f"{task}_{method}_ref_rank_histograms.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--method", default="npe_pfn")
    ap.add_argument(
        "--kde-compare-methods",
        nargs="+",
        default=None,
        help="Optional method list for a side-by-side KDE panel, e.g. nsf npe_pfn.",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--budgets", type=int, nargs="+", default=None)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument(
        "--obs", type=int, default=1, help="Reference observation for KDE overlay."
    )
    ap.add_argument("--hdr-masses", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--sample-dir", type=Path, default=SAMPLE_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(
        args.task, args.method, args.seeds, args.budgets, args.sample_dir
    )
    n_ref = min(args.n_ref, min(r.flow_samples.shape[0] for r in runs))
    if not 1 <= args.obs <= n_ref:
        raise ValueError(f"--obs must be between 1 and {n_ref}, got {args.obs}")
    print(
        f"Loaded {len(runs)} runs for task={args.task}, method={args.method}; "
        f"budgets={sorted({r.budget for r in runs})}, seeds={sorted({r.seed for r in runs})}"
    )
    refs = load_reference_samples(args.task, n_ref)

    moment_rows, global_rows = compute_metrics(runs, refs, args.hdr_masses)
    stem = f"{args.task}_{args.method}"
    write_csv(args.out_dir / f"{stem}_moment_metrics.csv", moment_rows)
    write_csv(args.out_dir / f"{stem}_global_metrics.csv", global_rows)

    plot_kde_budget_overlay(runs, refs, args.out_dir, args.obs)
    plot_moment_trajectories(moment_rows, args.task, args.method, args.out_dir)
    plot_whitened_diagnostics(global_rows, args.task, args.method, args.out_dir)
    plot_hdr_ratios(global_rows, args.task, args.method, args.hdr_masses, args.out_dir)
    plot_rank_histograms(moment_rows, args.task, args.method, args.out_dir)

    if args.kde_compare_methods is not None:
        method_runs = {
            method: discover_runs(
                args.task, method, args.seeds, args.budgets, args.sample_dir
            )
            for method in args.kde_compare_methods
        }
        method_moment_rows: dict[str, list[dict[str, float | int | str]]] = {}
        method_global_rows: dict[str, list[dict[str, float | int | str]]] = {}
        for method, compare_runs in method_runs.items():
            compare_moment_rows, compare_global_rows = compute_metrics(
                compare_runs, refs, args.hdr_masses
            )
            method_moment_rows[method] = compare_moment_rows
            method_global_rows[method] = compare_global_rows
        plot_kde_method_panel(method_runs, refs, args.out_dir, args.obs)
        plot_location_2d_kde_panel(method_runs, refs, args.out_dir, args.obs)
        plot_moment_method_panel(method_moment_rows, args.task, args.out_dir)
        plot_hdr_method_panel(
            method_global_rows, args.task, args.hdr_masses, args.out_dir
        )
        plot_moment_hdr_method_panel(
            method_moment_rows,
            method_global_rows,
            args.task,
            args.hdr_masses,
            args.out_dir,
        )


if __name__ == "__main__":
    main()
