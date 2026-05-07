"""Accessible summary plot for marginal quantile recovery across tasks.

The validation jobs compare probe-predicted marginal posterior quantiles
against empirical quantiles from reference posterior samples. This script
aggregates the best-layer validation outputs and produces a single horizontal
dot plot: one point per task, with the seed range shown as a line.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _tex_table import write_tex_tabular

VALIDATE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/quantile_validate")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")
CSV_OUT = Path("pfn_testing/sbi/outputs/layer_ablation/quantile_recovery_summary.csv")
TEX_OUT = Path("pfn_testing/sbi/outputs/layer_ablation/tables/quantile_recovery_pinball.tex")

EXCLUDED_TASK_SUFFIXES = ("_raw",)

TASK_LABELS = {
    "ar1_ts_t50": "AR(1) time series",
    "bernoulli_glm": "Bernoulli GLM",
    "bernoulli_glm_distractors": "Bernoulli GLM + distractors",
    "bernoulli_glm_raw": "Bernoulli GLM raw",
    "gaussian_linear": "Gaussian linear",
    "gaussian_linear_uniform": "Gaussian linear uniform",
    "gaussian_mixture": "Gaussian mixture",
    "gaussian_mixture_distractors": "Gaussian mixture + distractors",
    "lotka_volterra": "Lotka-Volterra",
    "lotka_volterra_raw": "Lotka-Volterra raw",
    "ou": "Ornstein-Uhlenbeck",
    "sir": "SIR",
    "sir_distractors": "SIR + distractors",
    "sir_raw": "SIR raw",
    "slcp": "SLCP",
    "slcp_distractors": "SLCP + distractors",
    "solar_dynamo": "Solar dynamo",
    "two_moons": "Two moons",
    "two_moons_distractors": "Two moons + distractors",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_runs() -> dict[str, list[dict[str, float | int]]]:
    runs: dict[str, list[dict[str, float | int]]] = {}
    for path in sorted(VALIDATE_DIR.glob("*_s*.npz")):
        data = dict(np.load(path, allow_pickle=True))
        task = str(data["task"])
        if task.endswith(EXCLUDED_TASK_SUFFIXES):
            continue
        best_layer = int(data["best_layer"])
        corr = np.asarray(data["corr"][best_layer], dtype=float)
        rmse = np.asarray(data["rmse"][best_layer], dtype=float)
        ref_pinball = pinball_loss(
            theta_true=np.asarray(data["theta_true"], dtype=float),
            quantiles=np.asarray(data["emp_q"], dtype=float),
            taus=np.asarray(data["taus"], dtype=float),
        )
        probe_pinball = float(data["pinball_ref"][best_layer])
        runs.setdefault(task, []).append(
            {
                "seed": int(data["seed"]),
                "best_layer": best_layer,
                "r_mean": float(np.nanmean(corr)),
                "r_median": float(corr[median_tau_index(data)]),
                "rmse_mean": float(np.nanmean(rmse)),
                "probe_pinball": probe_pinball,
                "reference_pinball": ref_pinball,
                "pinball_ratio": probe_pinball / ref_pinball if ref_pinball > 1e-12 else float("nan"),
            }
        )
    return runs


def pinball_loss(theta_true: np.ndarray, quantiles: np.ndarray, taus: np.ndarray) -> float:
    """Mean pinball loss for quantiles with shape (n_ref, n_tau, dim_theta)."""
    err = theta_true[:, None, :] - quantiles
    loss = np.maximum(taus[None, :, None] * err, (taus[None, :, None] - 1.0) * err)
    return float(loss.mean())


def median_tau_index(data: dict) -> int:
    taus = np.asarray(data["taus"], dtype=float)
    return int(np.argmin(np.abs(taus - 0.5)))


def summarize_runs(runs: dict[str, list[dict[str, float | int]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task, task_runs in runs.items():
        r_mean = np.asarray([run["r_mean"] for run in task_runs], dtype=float)
        rmse = np.asarray([run["rmse_mean"] for run in task_runs], dtype=float)
        probe_pinball = np.asarray([run["probe_pinball"] for run in task_runs], dtype=float)
        reference_pinball = np.asarray([run["reference_pinball"] for run in task_runs], dtype=float)
        pinball_ratio = np.asarray([run["pinball_ratio"] for run in task_runs], dtype=float)
        rows.append(
            {
                "task": task,
                "label": TASK_LABELS.get(task, task.replace("_", " ")),
                "n_seeds": len(task_runs),
                "mean_r": float(r_mean.mean()),
                "min_r": float(r_mean.min()),
                "max_r": float(r_mean.max()),
                "mean_rmse": float(rmse.mean()),
                "mean_probe_pinball": float(probe_pinball.mean()),
                "sd_probe_pinball": float(probe_pinball.std()),
                "mean_reference_pinball": float(reference_pinball.mean()),
                "mean_pinball_ratio": float(pinball_ratio.mean()),
                "sd_pinball_ratio": float(pinball_ratio.std()),
                "min_pinball_ratio": float(pinball_ratio.min()),
                "max_pinball_ratio": float(pinball_ratio.max()),
                "seeds": ",".join(
                    str(int(run["seed"]))
                    for run in sorted(task_runs, key=lambda r: r["seed"])
                ),
                "best_layers": ",".join(
                    str(int(run["best_layer"]))
                    for run in sorted(task_runs, key=lambda r: r["seed"])
                ),
            }
        )
    return sorted(rows, key=lambda row: float(row["mean_r"]))


def write_summary_csv(rows: list[dict[str, object]]) -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "label",
        "n_seeds",
        "mean_r",
        "min_r",
        "max_r",
        "mean_rmse",
        "mean_probe_pinball",
        "sd_probe_pinball",
        "mean_reference_pinball",
        "mean_pinball_ratio",
        "sd_pinball_ratio",
        "min_pinball_ratio",
        "max_pinball_ratio",
        "seeds",
        "best_layers",
    ]
    with CSV_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pm(mean: float, sd: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def write_pinball_table(rows: list[dict[str, object]]) -> None:
    table_rows = []
    for row in sorted(rows, key=lambda r: float(r["mean_pinball_ratio"])):
        table_rows.append(
            [
                str(row["label"]),
                f"{float(row['mean_r']):.3f}",
                fmt_pm(
                    float(row["mean_pinball_ratio"]),
                    float(row["sd_pinball_ratio"]),
                ),
                f"{float(row['mean_probe_pinball']):.4f}",
                f"{float(row['mean_reference_pinball']):.4f}",
            ]
        )
    write_tex_tabular(
        out_path=TEX_OUT,
        columns=[
            "Task",
            "Mean $r$",
            "Pinball ratio",
            "Probe pinball",
            "Ref. pinball",
        ],
        rows=table_rows,
        column_align="lrrrr",
        source_script="scripts/plot_quantile_recovery.py",
    )


def plot_recovery(rows: list[dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = [str(row["label"]) for row in rows]
    means = np.asarray([row["mean_r"] for row in rows], dtype=float)
    mins = np.asarray([row["min_r"] for row in rows], dtype=float)
    maxs = np.asarray([row["max_r"] for row in rows], dtype=float)
    ratios = np.asarray([row["mean_pinball_ratio"] for row in rows], dtype=float)
    ratio_mins = np.asarray([row["min_pinball_ratio"] for row in rows], dtype=float)
    ratio_maxs = np.asarray([row["max_pinball_ratio"] for row in rows], dtype=float)
    y = np.arange(len(rows))
    n_runs = sum(int(row["n_seeds"]) for row in rows)

    fig_height = max(6.5, 0.36 * len(rows) + 2.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, fig_height),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.08},
    )
    ax = axes[0]

    ax.axvspan(0.9, 1.0, color="#E7F0EA", zorder=0)
    ax.axvline(0.9, color="#6B7D6B", linestyle=":", linewidth=1.2)
    ax.hlines(y, mins, maxs, color="#747474", linewidth=2.0, alpha=0.75, zorder=2)

    colors = np.where(means >= 0.9, "#0B6E99", "#D55E00")
    ax.scatter(means, y, s=70, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    for yi, mean in zip(y, means, strict=True):
        ax.text(
            min(mean + 0.012, 0.985),
            yi,
            f"{mean:.2f}",
            va="center",
            ha="left" if mean < 0.97 else "right",
            fontsize=8.5,
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.6, 1.01)
    ax.set_xlabel("Correlation between predicted and reference marginal quantiles")
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(axis="y", length=0)
    ax.set_title("A. Quantile variation")

    ax = axes[1]
    ax.axvline(1.0, color="#6B7D6B", linestyle=":", linewidth=1.2)
    ax.hlines(y, ratio_mins, ratio_maxs, color="#747474", linewidth=2.0, alpha=0.75, zorder=2)
    ratio_colors = np.where(ratios <= 1.5, "#0B6E99", "#D55E00")
    ax.scatter(ratios, y, s=70, color=ratio_colors, edgecolor="white", linewidth=0.8, zorder=3)
    for yi, ratio in zip(y, ratios, strict=True):
        ax.text(
            ratio + 0.10,
            yi,
            f"{ratio:.1f}x",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#222222",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0.8, max(2.0, float(np.nanmax(ratio_maxs)) * 1.18))
    ax.set_xlabel("Pinball loss ratio")
    ax.set_title("B. Absolute calibration")
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(axis="y", length=0)
    ax.text(1.03, len(rows) - 0.45, "Reference target", fontsize=9, color="#3F5E46")

    subtitle = (
        f"{len(rows)} tasks, {n_runs} validation runs. "
        "Raw-task variants excluded; dots are task means and lines show seed range."
    )
    fig.suptitle("Marginal quantile recovery across problems", fontsize=13, y=0.985)
    fig.text(0.5, 0.945, subtitle, fontsize=9.5, color="#444444", ha="center")
    axes[0].text(0.905, len(rows) - 0.45, "High correlation", fontsize=9, color="#3F5E46")

    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.09, top=0.82, wspace=0.08)
    out_png = FIG_DIR / "quantile_recovery_accessible.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No quantile validation outputs found in {VALIDATE_DIR}")

    rows = summarize_runs(runs)
    write_summary_csv(rows)
    write_pinball_table(rows)
    plot_recovery(rows)
    print(f"Wrote {CSV_OUT}")

    print("\n=== Quantile recovery summary ===")
    print(f"{'task':<36} {'n':>2} {'mean_r':>7} {'r_range':>15} {'pin_ratio':>10}")
    for row in reversed(rows):
        print(
            f"{row['task']:<36} {row['n_seeds']:>2} "
            f"{row['mean_r']:>7.3f} "
            f"[{row['min_r']:.3f}, {row['max_r']:.3f}] "
            f"{row['mean_pinball_ratio']:>10.2f}"
        )


if __name__ == "__main__":
    main()
