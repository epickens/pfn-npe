"""Appendix table for PFN-embedding vs raw-x quantile probes.

The table uses matched task/seed validation outputs from
`quantile_validate/` and `quantile_validate_raw/`. It reports mean quantile
correlation and pinball-loss ratio across seeds. Raw task variants are excluded
because they are not part of the paper's posterior-sample benchmark figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _tex_table import write_tex_tabular  # noqa: E402

PFN_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/quantile_validate")
RAW_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/quantile_validate_raw")
OUT_TEX = Path("pfn_testing/sbi/outputs/layer_ablation/tables/quantile_raw_ablation.tex")
OUT_CSV = Path("pfn_testing/sbi/outputs/layer_ablation/quantile_raw_ablation.csv")

TASK_LABELS = {
    "two_moons": "Two moons",
    "gaussian_mixture": "Gaussian mixture",
    "gaussian_linear": "Gaussian linear",
    "gaussian_linear_uniform": "Gaussian linear uniform",
    "bernoulli_glm": "Bernoulli GLM",
    "sir": "SIR",
    "lotka_volterra": "Lotka-Volterra",
    "slcp": "SLCP",
    "slcp_distractors": "SLCP + distractors",
    "two_moons_distractors": "Two moons + distractors",
    "gaussian_mixture_distractors": "Gaussian mixture + distractors",
    "bernoulli_glm_distractors": "Bernoulli GLM + distractors",
    "sir_distractors": "SIR + distractors",
    "ar1_ts_t50": "AR(1) time series",
    "ou": "Ornstein-Uhlenbeck",
    "solar_dynamo": "Solar dynamo",
}
TASK_ORDER = list(TASK_LABELS)


def pinball_loss(theta_true: np.ndarray, quantiles: np.ndarray, taus: np.ndarray) -> float:
    err = theta_true[:, None, :] - quantiles
    loss = np.maximum(taus[None, :, None] * err, (taus[None, :, None] - 1.0) * err)
    return float(loss.mean())


def load_run(path: Path) -> dict[str, float | int | str]:
    data = dict(np.load(path, allow_pickle=True))
    task = str(data["task"])
    best = int(data["best_layer"])
    corr = np.asarray(data["corr"][best], dtype=float)
    taus = np.asarray(data["taus"], dtype=float)
    reference_pinball = pinball_loss(
        theta_true=np.asarray(data["theta_true"], dtype=float),
        quantiles=np.asarray(data["emp_q"], dtype=float),
        taus=taus,
    )
    probe_pinball = float(np.asarray(data["pinball_ref"], dtype=float)[best])
    return {
        "task": task,
        "seed": int(data["seed"]),
        "corr": float(np.nanmean(corr)),
        "probe_pinball": probe_pinball,
        "reference_pinball": reference_pinball,
        "pinball_ratio": probe_pinball / reference_pinball,
    }


def load_runs(base: Path) -> dict[tuple[str, int], dict[str, float | int | str]]:
    runs = {}
    for path in sorted(base.glob("*_s*.npz")):
        run = load_run(path)
        task = str(run["task"])
        if task.endswith("_raw"):
            continue
        if task not in TASK_LABELS:
            continue
        runs[(task, int(run["seed"]))] = run
    return runs


def mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def fmt_mean_sd(mean: float, sd: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def main() -> None:
    pfn = load_runs(PFN_DIR)
    raw = load_runs(RAW_DIR)
    rows: list[list[str]] = []
    csv_rows: list[list[str]] = []

    for task in TASK_ORDER:
        seeds = sorted(seed for t, seed in (set(pfn) & set(raw)) if t == task)
        if not seeds:
            continue

        pfn_corr, raw_corr = [], []
        pfn_ratio, raw_ratio = [], []
        for seed in seeds:
            pfn_run = pfn[(task, seed)]
            raw_run = raw[(task, seed)]
            pfn_corr.append(float(pfn_run["corr"]))
            raw_corr.append(float(raw_run["corr"]))
            pfn_ratio.append(float(pfn_run["pinball_ratio"]))
            raw_ratio.append(float(raw_run["pinball_ratio"]))

        pfn_corr_m, pfn_corr_s = mean_sd(pfn_corr)
        raw_corr_m, raw_corr_s = mean_sd(raw_corr)
        pfn_ratio_m, pfn_ratio_s = mean_sd(pfn_ratio)
        raw_ratio_m, raw_ratio_s = mean_sd(raw_ratio)

        rows.append(
            [
                TASK_LABELS[task],
                str(len(seeds)),
                fmt_mean_sd(pfn_corr_m, pfn_corr_s, digits=3),
                fmt_mean_sd(raw_corr_m, raw_corr_s, digits=3),
                fmt_mean_sd(pfn_ratio_m, pfn_ratio_s),
                fmt_mean_sd(raw_ratio_m, raw_ratio_s),
            ]
        )
        csv_rows.append(
            [
                task,
                str(len(seeds)),
                f"{pfn_corr_m:.6f}",
                f"{pfn_corr_s:.6f}",
                f"{raw_corr_m:.6f}",
                f"{raw_corr_s:.6f}",
                f"{pfn_ratio_m:.6f}",
                f"{pfn_ratio_s:.6f}",
                f"{raw_ratio_m:.6f}",
                f"{raw_ratio_s:.6f}",
            ]
        )

    if not rows:
        raise SystemExit("No matched PFN/raw quantile validation outputs found.")

    write_tex_tabular(
        out_path=OUT_TEX,
        columns=[
            "Task",
            "$n$",
            "PFN $r$",
            "Raw $x$ $r$",
            "PFN pinball ratio",
            "Raw $x$ pinball ratio",
        ],
        rows=rows,
        column_align="lrrrrr",
        source_script="scripts/quantile_raw_ablation_table.py",
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text(
        "\n".join(
            [
                "task,n,pfn_corr_mean,pfn_corr_sd,raw_corr_mean,raw_corr_sd,"
                "pfn_ratio_mean,pfn_ratio_sd,raw_ratio_mean,raw_ratio_sd",
                *(",".join(row) for row in csv_rows),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_CSV}")

    print("\n=== PFN vs raw-x quantile ablation ===")
    print(f"{'task':<34} {'n':>2} {'pfn_r':>7} {'raw_r':>7} {'pfn_ratio':>10} {'raw_ratio':>10}")
    for row, csv_row in zip(rows, csv_rows, strict=True):
        print(
            f"{csv_row[0]:<34} {csv_row[1]:>2} "
            f"{float(csv_row[2]):>7.3f} {float(csv_row[4]):>7.3f} "
            f"{float(csv_row[6]):>10.2f} {float(csv_row[8]):>10.2f}"
        )


if __name__ == "__main__":
    main()
