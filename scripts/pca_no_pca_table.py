"""PCA vs no-PCA PFN-NPE appendix table.

Compares the canonical PCA-64 PFN-NPE C2ST decomposition (`nsf`) against
the explicit no-PCA reruns (`nsf_no_pca`). The layer-ablation filename
convention treats missing `_n{budget}` suffixes as 10k, matching
`aggregate_c2st_table.py` and the manuscript's `c2st_joint.tex`. The table
reports the broad 10k comparison plus the 50k core-task comparison once the
P27 no-PCA budget expansion has completed.

Inputs:
  - pfn_testing/sbi/outputs/layer_ablation/c2st_decomp/{task}_s{seed}_nsf.npz
  - pfn_testing/sbi/outputs/layer_ablation/c2st_decomp/{task}_s{seed}_nsf_no_pca.npz

Usage:
  uv run python scripts/pca_no_pca_table.py \
      --tex-out results/tables/pca_no_pca_budget.tex
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tex_table import write_tex_tabular  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DECOMP_DIR = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/c2st_decomp"
DEFAULT_CSV = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/pca_no_pca_budget.csv"

SEEDS = (42, 7, 123)
CORE_TASKS = [
    ("two_moons", "Two moons"),
    ("bernoulli_glm", "Bernoulli GLM"),
    ("slcp", "SLCP"),
    ("gaussian_mixture_distractors", "G. mixture + distractors"),
    ("ar1_ts_t50", "AR(1), T=50"),
]
TASKS_10K = [
    ("two_moons", "Two moons"),
    ("gaussian_mixture", "Gaussian mixture"),
    ("gaussian_linear", "Gaussian linear"),
    ("gaussian_linear_uniform", "Gaussian linear uniform"),
    ("bernoulli_glm", "Bernoulli GLM"),
    ("slcp", "SLCP"),
    ("sir", "SIR"),
    ("lotka_volterra", "Lotka-Volterra"),
    ("gaussian_mixture_distractors", "G. mixture + distractors"),
    ("two_moons_distractors", "Two moons + distractors"),
    ("bernoulli_glm_distractors", "Bernoulli GLM + distractors"),
    ("sir_distractors", "SIR + distractors"),
    ("ar1_ts_t50", "AR(1), T=50"),
    ("ou", "OU"),
    ("solar_dynamo", "Solar dynamo"),
]
BUDGET_BLOCKS = [
    (10000, TASKS_10K),
    (50000, CORE_TASKS),
]


def _method_suffix(method: str, budget: int) -> str:
    return method if budget == 10000 else f"{method}_n{budget}"


def _method_path(task: str, seed: int, method: str, budget: int) -> Path:
    return DECOMP_DIR / f"{task}_s{seed}_{_method_suffix(method, budget)}.npz"


def _seed_mean(task: str, seed: int, method: str, budget: int) -> float:
    path = _method_path(task, seed, method, budget)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return float(np.mean(data["joint"]))


def _paired_values(task: str, budget: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    seeds = tuple(
        seed for seed in SEEDS
        if _method_path(task, seed, "nsf", budget).exists()
        and _method_path(task, seed, "nsf_no_pca", budget).exists()
    )
    if not seeds:
        raise FileNotFoundError(f"No paired nsf/nsf_no_pca seeds for {task} at {budget}")
    pca = np.array([_seed_mean(task, seed, "nsf", budget) for seed in seeds])
    no_pca = np.array([_seed_mean(task, seed, "nsf_no_pca", budget) for seed in seeds])
    return pca, no_pca, seeds


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
    return mean, std


def _fmt(mean: float, std: float, bold: bool = False, decimals: int = 3) -> str:
    mean_text = f"{mean:.{decimals}f}"
    if bold:
        mean_text = rf"\mathbf{{{mean_text}}}"
    return rf"${mean_text}\,\pm\,{std:.2f}$"


def _fmt_delta(delta: float, decimals: int = 3) -> str:
    if abs(delta) < 0.5 * 10 ** (-decimals):
        delta = 0.0
    return f"{delta:+.{decimals}f}"


def _fmt_budget(budget: int) -> str:
    if budget == 10000:
        return r"$10^4$"
    if budget == 50000:
        return r"$5{\times}10^4$"
    if budget == 100000:
        return r"$10^5$"
    return str(budget)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    tex_rows: list[list[str]] = []
    midrules_after: list[int] = []
    for budget, task_list in BUDGET_BLOCKS:
        for task, label in task_list:
            pca_values, no_pca_values, seeds = _paired_values(task, budget)
            pca, pca_std = _mean_std(pca_values)
            no_pca, no_pca_std = _mean_std(no_pca_values)
            deltas = no_pca_values - pca_values
            delta, delta_std = _mean_std(deltas)
            rows.append(
                {
                    "budget": budget,
                    "task": task,
                    "label": label,
                    "seeds": " ".join(str(s) for s in seeds),
                    "n_seeds": len(seeds),
                    "pca64_joint_mean": pca,
                    "pca64_joint_std": pca_std,
                    "no_pca_joint_mean": no_pca,
                    "no_pca_joint_std": no_pca_std,
                    "delta_no_pca_minus_pca": delta,
                    "delta_std": delta_std,
                }
            )
            pca_best = pca <= no_pca
            no_pca_best = no_pca < pca
            tex_rows.append(
                [
                    _fmt_budget(budget),
                    label,
                    _fmt(pca, pca_std, pca_best, args.decimals),
                    _fmt(no_pca, no_pca_std, no_pca_best, args.decimals),
                    _fmt_delta(delta, args.decimals),
                ]
            )
        midrules_after.append(len(tex_rows) - 1)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.csv_out}")

    print(
        f"{'Budget':>8} {'Task':<30} {'PCA-64':>12} "
        f"{'No PCA':>12} {'Delta':>9} {'Seeds':>10}"
    )
    for row in rows:
        print(
            f"{row['budget']:>8} "
            f"{row['label']:<30} "
            f"{row['pca64_joint_mean']:>6.3f}±{row['pca64_joint_std']:<5.2f} "
            f"{row['no_pca_joint_mean']:>6.3f}±{row['no_pca_joint_std']:<5.2f} "
            f"{row['delta_no_pca_minus_pca']:>+9.3f} "
            f"{row['seeds']:>10}"
        )

    if args.tex_out is not None:
        write_tex_tabular(
            out_path=args.tex_out,
            columns=[
                "Budget",
                "Task",
                "PCA-64",
                "No PCA",
                r"$\Delta$",
            ],
            rows=tex_rows,
            column_align="llccc",
            midrules_after=midrules_after,
            source_script="scripts/pca_no_pca_table.py",
        )


if __name__ == "__main__":
    main()
