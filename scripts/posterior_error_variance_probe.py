"""Probe how posterior spread and moment errors relate to C2ST performance.

This is a lightweight post-hoc diagnostic. It uses saved posterior samples from
``layer_ablation/flow_vs_quantile`` and saved per-observation C2ST decompositions
from ``layer_ablation/c2st_decomp``. For each reference observation it compares
model samples to SBIBM reference posterior samples and asks which errors track
downstream C2ST:

* intrinsic posterior spread / anisotropy,
* marginal variance-scale error,
* posterior mean error,
* covariance and correlation error.

Outputs:
  - ``posterior_error_variance_probe.csv``: one row per task/seed/observation.
  - ``posterior_error_variance_probe_correlations.csv``: correlation summaries.
  - ``figures/posterior_error_variance_probe.png``: scatter diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task  # noqa: E402


EPS = 1e-8
DEFAULT_CMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
DEFAULT_C2ST_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
DEFAULT_OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation")
FLOW_FILE_RE = re.compile(r"(?P<task>.+)_s(?P<seed>\d+)(?:_(?P<method>.+))?\.npz$")


PREDICTORS = [
    "ref_log_total_var",
    "ref_var_anisotropy_cv",
    "abs_log_var_error",
    "abs_log_total_var_bias",
    "mean_rmse_std",
    "cov_rel_fro",
    "corr_offdiag_mae",
    "quantile_rmse_std",
]
OUTCOMES = ["joint_c2st", "marginal_c2st", "rank_c2st"]


def covariance(samples: np.ndarray) -> np.ndarray:
    cov = np.cov(np.asarray(samples, dtype=np.float64), rowvar=False)
    cov = np.atleast_2d(cov)
    return np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)


def corrcoef_safe(samples: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(np.asarray(samples, dtype=np.float64), rowvar=False)
    corr = np.atleast_2d(corr)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def offdiag(mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] < 2:
        return np.array([], dtype=np.float64)
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def effective_dim(cov: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    eig = np.clip(eig, 0.0, None)
    denom = float(np.sum(eig**2))
    if denom <= EPS:
        return float("nan")
    return float(np.sum(eig) ** 2 / denom)


def matched_files(
    cmp_dir: Path,
    c2st_dir: Path,
    tasks: set[str] | None,
    method: str | None,
) -> Iterable[tuple[str, int, str, Path, Path]]:
    for flow_path in sorted(cmp_dir.glob("*.npz")):
        match = FLOW_FILE_RE.match(flow_path.name)
        if not match:
            continue
        task = match.group("task")
        seed = int(match.group("seed"))
        file_method = match.group("method") or ""
        if tasks is not None and task not in tasks:
            continue
        if method is not None and file_method != method:
            continue
        c2st_path = c2st_dir / flow_path.name
        if c2st_path.exists():
            yield task, seed, file_method, flow_path, c2st_path


def reference_samples(task_name: str, n_obs: int) -> list[np.ndarray]:
    task = get_task(task_name)
    refs = []
    for obs_idx in range(1, n_obs + 1):
        refs.append(
            task.get_reference_posterior_samples(num_observation=obs_idx)
            .numpy()
            .astype(np.float64)
        )
    return refs


def observation_row(
    task: str,
    seed: int,
    obs_idx: int,
    model_samples: np.ndarray,
    ref_samples: np.ndarray,
    flow_q: np.ndarray,
    emp_q: np.ndarray,
    c2st: dict[str, np.ndarray],
) -> dict[str, float | int | str]:
    n = min(len(model_samples), len(ref_samples))
    model = np.asarray(model_samples[:n], dtype=np.float64)
    ref = np.asarray(ref_samples[:n], dtype=np.float64)

    ref_mean = ref.mean(axis=0)
    model_mean = model.mean(axis=0)
    ref_var = ref.var(axis=0) + EPS
    model_var = model.var(axis=0) + EPS
    ref_cov = covariance(ref)
    model_cov = covariance(model)
    ref_corr = corrcoef_safe(ref)
    model_corr = corrcoef_safe(model)

    log_var_ratio = np.log(model_var) - np.log(ref_var)
    mean_err = model_mean - ref_mean
    ref_total_var = float(np.mean(ref_var))
    model_total_var = float(np.mean(model_var))
    cov_denom = float(np.linalg.norm(ref_cov, ord="fro")) + EPS

    corr_ref_off = offdiag(ref_corr)
    corr_model_off = offdiag(model_corr)
    if corr_ref_off.size:
        corr_mae = float(np.mean(np.abs(corr_model_off - corr_ref_off)))
    else:
        corr_mae = float("nan")

    q_err = np.asarray(flow_q, dtype=np.float64) - np.asarray(emp_q, dtype=np.float64)
    q_scale = np.sqrt(ref_var)[None, :]
    quantile_rmse_std = float(np.sqrt(np.mean((q_err / q_scale) ** 2)))

    return {
        "task": task,
        "seed": seed,
        "obs": obs_idx,
        "joint_c2st": float(c2st["joint"][obs_idx - 1]),
        "marginal_c2st": float(c2st["marginal"][obs_idx - 1]),
        "rank_c2st": float(c2st["rank"][obs_idx - 1]),
        "ref_total_var": ref_total_var,
        "ref_log_total_var": float(np.log(ref_total_var + EPS)),
        "model_total_var": model_total_var,
        "signed_log_total_var_bias": float(
            np.log(model_total_var + EPS) - np.log(ref_total_var + EPS)
        ),
        "abs_log_total_var_bias": float(
            abs(np.log(model_total_var + EPS) - np.log(ref_total_var + EPS))
        ),
        "ref_var_anisotropy_cv": float(np.std(ref_var) / (np.mean(ref_var) + EPS)),
        "model_var_anisotropy_cv": float(
            np.std(model_var) / (np.mean(model_var) + EPS)
        ),
        "ref_effective_dim": effective_dim(ref_cov),
        "model_effective_dim": effective_dim(model_cov),
        "signed_log_var_error": float(np.mean(log_var_ratio)),
        "abs_log_var_error": float(np.mean(np.abs(log_var_ratio))),
        "rel_var_rmse": float(
            np.sqrt(np.mean(((model_var - ref_var) / ref_var) ** 2))
        ),
        "mean_rmse": float(np.sqrt(np.mean(mean_err**2))),
        "mean_rmse_std": float(np.sqrt(np.mean(mean_err**2 / ref_var))),
        "cov_rel_fro": float(np.linalg.norm(model_cov - ref_cov, ord="fro") / cov_denom),
        "corr_offdiag_mae": corr_mae,
        "quantile_rmse_std": quantile_rmse_std,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def finite_pair(rows: list[dict[str, object]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    x = to_float_array(rows, x_key)
    y = to_float_array(rows, y_key)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def corr_summary(
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    level: str,
    method: str,
) -> dict[str, object]:
    x, y = finite_pair(rows, x_key, y_key)
    if len(x) < 3 or np.std(x) <= EPS or np.std(y) <= EPS:
        return {
            "level": level,
            "method": method,
            "predictor": x_key,
            "outcome": y_key,
            "n": len(x),
            "r": float("nan"),
            "p": float("nan"),
        }
    if method == "spearman":
        r, p = spearmanr(x, y)
    elif method == "pearson":
        r, p = pearsonr(x, y)
    else:
        raise ValueError(method)
    return {
        "level": level,
        "method": method,
        "predictor": x_key,
        "outcome": y_key,
        "n": len(x),
        "r": float(r),
        "p": float(p),
    }


def aggregate_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)

    out = []
    numeric = [
        key
        for key, value in rows[0].items()
        if key not in {"task", "seed", "obs"} and isinstance(value, (float, int))
    ]
    for group_key, group_rows in grouped.items():
        row = {key: value for key, value in zip(keys, group_key, strict=False)}
        for key in numeric:
            values = np.asarray([float(r[key]) for r in group_rows], dtype=np.float64)
            row[key] = float(np.nanmean(values))
        row["n_rows"] = len(group_rows)
        out.append(row)
    return out


def within_group_residual_rows(
    rows: list[dict[str, object]],
    group_key: str,
    numeric_keys: list[str],
) -> list[dict[str, object]]:
    grouped: dict[object, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[row[group_key]].append(idx)

    residual_rows = [dict(row) for row in rows]
    for key in numeric_keys:
        values = to_float_array(rows, key)
        residuals = np.full_like(values, np.nan, dtype=np.float64)
        for indices in grouped.values():
            group_values = values[indices]
            mask = np.isfinite(group_values)
            if mask.sum() == 0:
                continue
            residuals[indices] = group_values - np.nanmean(group_values)
        for idx, value in enumerate(residuals):
            residual_rows[idx][key] = float(value)
    return residual_rows


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + EPS)


def regression_r2(rows: list[dict[str, object]], predictors: list[str], outcome: str) -> dict[str, object]:
    arrays = [to_float_array(rows, key) for key in predictors]
    y = to_float_array(rows, outcome)
    mask = np.isfinite(y)
    for arr in arrays:
        mask &= np.isfinite(arr)
    y = y[mask]
    if len(y) <= len(predictors) + 2 or np.nanstd(y) <= EPS:
        return {"outcome": outcome, "predictors": ",".join(predictors), "n": len(y), "r2": float("nan")}
    x_mat = np.column_stack([zscore(arr[mask]) for arr in arrays])
    x_design = np.column_stack([np.ones(len(y)), x_mat])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    pred = x_design @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    out: dict[str, object] = {
        "outcome": outcome,
        "predictors": ",".join(predictors),
        "n": len(y),
        "r2": float(1.0 - ss_res / (ss_tot + EPS)),
    }
    for key, coef in zip(predictors, beta[1:], strict=False):
        out[f"beta_{key}"] = float(coef)
    return out


def build_correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    corr_rows: list[dict[str, object]] = []
    task_seed_rows = aggregate_rows(rows, ("task", "seed"))
    task_rows = aggregate_rows(rows, ("task",))
    numeric_keys = PREDICTORS + OUTCOMES
    within_task_rows = within_group_residual_rows(rows, "task", numeric_keys)

    for outcome in OUTCOMES:
        for predictor in PREDICTORS:
            corr_rows.append(corr_summary(rows, predictor, outcome, "obs", "spearman"))
            corr_rows.append(corr_summary(task_seed_rows, predictor, outcome, "task_seed_mean", "spearman"))
            corr_rows.append(corr_summary(task_rows, predictor, outcome, "task_mean", "spearman"))
            corr_rows.append(corr_summary(within_task_rows, predictor, outcome, "within_task_obs", "pearson"))

    for predictor_set in [
        ["ref_log_total_var"],
        ["abs_log_var_error"],
        ["mean_rmse_std"],
        ["cov_rel_fro"],
        ["corr_offdiag_mae"],
        ["mean_rmse_std", "abs_log_var_error", "corr_offdiag_mae"],
        ["mean_rmse_std", "abs_log_var_error", "cov_rel_fro"],
    ]:
        for outcome in OUTCOMES:
            row = regression_r2(rows, predictor_set, outcome)
            row.update({"level": "obs", "method": "ols_r2", "predictor": row["predictors"], "r": row["r2"], "p": float("nan")})
            corr_rows.append(row)
    return corr_rows


def plot_probe(rows: list[dict[str, object]], out_path: Path, title_suffix: str) -> None:
    task_names = sorted({str(r["task"]) for r in rows})
    colors = dict(zip(task_names, plt.cm.tab10(np.linspace(0, 1, len(task_names))), strict=False))

    panels = [
        ("abs_log_var_error", "joint_c2st", "Variance-scale error", "Joint C2ST"),
        ("mean_rmse_std", "joint_c2st", "Mean error / posterior sd", "Joint C2ST"),
        ("corr_offdiag_mae", "rank_c2st", "Correlation MAE", "Rank C2ST"),
        ("ref_log_total_var", "joint_c2st", "Intrinsic log posterior variance", "Joint C2ST"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for ax, (x_key, y_key, x_label, y_label) in zip(axes.ravel(), panels, strict=False):
        for task in task_names:
            subset = [r for r in rows if r["task"] == task]
            x, y = finite_pair(subset, x_key, y_key)
            if len(x) == 0:
                continue
            ax.scatter(x, y, s=26, alpha=0.72, color=colors[task], label=task)
        all_x, all_y = finite_pair(rows, x_key, y_key)
        if len(all_x) >= 3 and np.std(all_x) > EPS:
            r, p = spearmanr(all_x, all_y)
            ax.text(
                0.03,
                0.96,
                f"Spearman r={r:+.2f}, p={p:.2g}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle(
        f"Posterior variance/error vs C2ST performance ({title_suffix})",
        y=0.985,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")


def print_key_findings(corr_rows: list[dict[str, object]]) -> None:
    print("\nTop observation-level Spearman correlations with joint C2ST")
    rows = [
        r for r in corr_rows
        if r.get("level") == "obs" and r.get("method") == "spearman"
        and r.get("outcome") == "joint_c2st" and math.isfinite(float(r.get("r", float("nan"))))
    ]
    for row in sorted(rows, key=lambda r: abs(float(r["r"])), reverse=True)[:8]:
        print(
            f"  {row['predictor']:<24} r={float(row['r']):+0.3f} "
            f"p={float(row['p']):.3g} n={row['n']}"
        )

    print("\nWithin-task residual Pearson correlations with joint C2ST")
    rows = [
        r for r in corr_rows
        if r.get("level") == "within_task_obs" and r.get("method") == "pearson"
        and r.get("outcome") == "joint_c2st" and math.isfinite(float(r.get("r", float("nan"))))
    ]
    for row in sorted(rows, key=lambda r: abs(float(r["r"])), reverse=True)[:8]:
        print(
            f"  {row['predictor']:<24} r={float(row['r']):+0.3f} "
            f"p={float(row['p']):.3g} n={row['n']}"
        )

    print("\nSelected OLS R2 for joint C2ST")
    rows = [
        r for r in corr_rows
        if r.get("method") == "ols_r2" and r.get("outcome") == "joint_c2st"
    ]
    for row in rows:
        print(f"  {row['predictors']:<54} R2={float(row['r2']):.3f} n={row['n']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmp-dir", type=Path, default=DEFAULT_CMP_DIR)
    parser.add_argument("--c2st-dir", type=Path, default=DEFAULT_C2ST_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task filter, e.g. --tasks slcp two_moons ar1_ts_t50.",
    )
    parser.add_argument(
        "--method",
        default="raw_tabq_ps1000",
        help="flow_vs_quantile method suffix to analyze. Use 'nsf_pca64' for "
             "rerun TabPFN-NPE PCA-64 artifacts. Pass 'ALL' for no method filter.",
    )
    parser.add_argument(
        "--out-stem",
        default=None,
        help="Output filename stem. Defaults to posterior_error_variance_probe "
             "for raw_tabq_ps1000, otherwise posterior_error_variance_probe_<method>.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    task_filter = set(args.tasks) if args.tasks else None
    method_filter = None if args.method == "ALL" else args.method
    files = list(matched_files(args.cmp_dir, args.c2st_dir, task_filter, method_filter))
    if not files:
        raise SystemExit(
            "No matched flow_vs_quantile and c2st_decomp files found. "
            "Run train_and_sample_flow.py first, then run_c2st_sweep.py for the "
            "same method suffix."
        )

    for task_name, seed, method, flow_path, c2st_path in files:
        print(f"Loading {task_name} seed={seed} method={method or 'default'}")
        flow_data = np.load(flow_path, allow_pickle=True)
        c2st_data = np.load(c2st_path, allow_pickle=True)
        flow_samples = np.asarray(flow_data["flow_samples"], dtype=np.float64)
        flow_q = np.asarray(flow_data["flow_q"], dtype=np.float64)
        emp_q = np.asarray(flow_data["emp_q"], dtype=np.float64)
        n_obs = flow_samples.shape[0]
        refs = reference_samples(task_name, n_obs)
        c2st = {
            "joint": np.asarray(c2st_data["joint"], dtype=np.float64),
            "marginal": np.asarray(c2st_data["marginal"], dtype=np.float64),
            "rank": np.asarray(c2st_data["rank"], dtype=np.float64),
        }
        for obs_idx in range(1, n_obs + 1):
            rows.append(
                observation_row(
                    task_name,
                    seed,
                    obs_idx,
                    flow_samples[obs_idx - 1],
                    refs[obs_idx - 1],
                    flow_q[obs_idx - 1],
                    emp_q[obs_idx - 1],
                    c2st,
                )
            )

    out_csv = args.out_dir / "posterior_error_variance_probe.csv"
    if args.out_stem is not None:
        out_stem = args.out_stem
    elif args.method == "raw_tabq_ps1000":
        out_stem = "posterior_error_variance_probe"
    else:
        out_stem = f"posterior_error_variance_probe_{args.method}"
    out_csv = args.out_dir / f"{out_stem}.csv"
    corr_csv = args.out_dir / f"{out_stem}_correlations.csv"
    fig_path = args.out_dir / f"figures/{out_stem}.png"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corr_rows = build_correlation_rows(rows)
    write_csv(out_csv, rows)
    write_csv(corr_csv, corr_rows)
    plot_probe(rows, fig_path, args.method)

    print(f"\nWrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {corr_csv} ({len(corr_rows)} rows)")
    print(f"Wrote {fig_path}")
    print_key_findings(corr_rows)


if __name__ == "__main__":
    main()
