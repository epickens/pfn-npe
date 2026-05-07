"""XL no-PCA flow-head diagnostic table.

Compares the default PCA-64 NSF readout (`nsf`), the standard NSF readout on
full embeddings (`nsf_no_pca`), and the XL NSF readout on full embeddings
(`nsf_xl_no_pca`) on the hard-task subset at 10k and 50k simulations.
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
DEFAULT_CSV = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/xl_no_pca_table.csv"
DEFAULT_TEX = REPO_ROOT / "results/tables/xl_no_pca_table.tex"

SEEDS = (42, 7, 123)
TASKS = [
    ("slcp", "SLCP"),
    ("lotka_volterra", "Lotka-Volterra"),
    ("ou", "OU"),
    ("ar1_ts_t50", "AR(1), T=50"),
    ("solar_dynamo", "Solar dynamo"),
]
BUDGETS = (10_000, 50_000)
METHODS = [
    ("nsf", "PCA-64 NSF"),
    ("nsf_no_pca", "No-PCA NSF"),
    ("nsf_xl_no_pca", "XL no-PCA NSF"),
]


def _method_suffix(method: str, budget: int) -> str:
    return method if budget == 10_000 else f"{method}_n{budget}"


def _path(task: str, seed: int, method: str, budget: int) -> Path:
    return DECOMP_DIR / f"{task}_s{seed}_{_method_suffix(method, budget)}.npz"


def _seed_mean(task: str, seed: int, method: str, budget: int) -> float | None:
    path = _path(task, seed, method, budget)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return float(np.mean(data["joint"]))


def _values(task: str, method: str, budget: int) -> np.ndarray:
    vals = [
        value
        for seed in SEEDS
        if (value := _seed_mean(task, seed, method, budget)) is not None
    ]
    return np.asarray(vals, dtype=float)


def _mean_std(values: np.ndarray) -> tuple[float | None, float | None]:
    if len(values) == 0:
        return None, None
    return (
        float(values.mean()),
        float(values.std(ddof=0)) if len(values) > 1 else 0.0,
    )


def _paired_delta(task: str, budget: int) -> float | None:
    deltas: list[float] = []
    for seed in SEEDS:
        base = _seed_mean(task, seed, "nsf", budget)
        xl = _seed_mean(task, seed, "nsf_xl_no_pca", budget)
        if base is not None and xl is not None:
            deltas.append(xl - base)
    if not deltas:
        return None
    return float(np.mean(deltas))


def _fmt_budget(budget: int) -> str:
    if budget == 10_000:
        return r"$10^4$"
    if budget == 50_000:
        return r"$5{\times}10^4$"
    return str(budget)


def _fmt_cell(mean: float | None, std: float | None, *, bold: bool, decimals: int) -> str:
    if mean is None or std is None:
        return "---"
    mean_text = f"{mean:.{decimals}f}"
    if bold:
        mean_text = rf"\mathbf{{{mean_text}}}"
    return rf"${mean_text}\,\pm\,{std:.2f}$"


def _fmt_delta(delta: float | None, decimals: int) -> str:
    if delta is None:
        return "---"
    if abs(delta) < 0.5 * 10 ** (-decimals):
        delta = 0.0
    return f"{delta:+.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tex-out", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args()

    csv_rows: list[dict[str, object]] = []
    tex_rows: list[list[str]] = []
    midrules_after: list[int] = []
    for budget in BUDGETS:
        for task, label in TASKS:
            method_stats: dict[str, tuple[float | None, float | None, int]] = {}
            for method, _method_label in METHODS:
                values = _values(task, method, budget)
                mean, std = _mean_std(values)
                method_stats[method] = (mean, std, len(values))
                csv_rows.append(
                    {
                        "budget": budget,
                        "task": task,
                        "method": method,
                        "mean": "" if mean is None else mean,
                        "std": "" if std is None else std,
                        "n_seeds": len(values),
                    }
                )

            finite_means = [
                mean for mean, _std, _n in method_stats.values()
                if mean is not None
            ]
            best = min(finite_means) if finite_means else None
            row = [_fmt_budget(budget), label]
            for method, _method_label in METHODS:
                mean, std, _n = method_stats[method]
                row.append(
                    _fmt_cell(
                        mean,
                        std,
                        bold=best is not None and mean == best,
                        decimals=args.decimals,
                    )
                )
            row.append(_fmt_delta(_paired_delta(task, budget), args.decimals))
            tex_rows.append(row)
        midrules_after.append(len(tex_rows) - 1)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {args.csv_out}")

    write_tex_tabular(
        out_path=args.tex_out,
        columns=[
            "Budget",
            "Task",
            "PCA-64 NSF",
            "No-PCA NSF",
            "XL no-PCA NSF",
            r"$\Delta_{\mathrm{XL-PCA}}$",
        ],
        rows=tex_rows,
        column_align="llcccc",
        midrules_after=midrules_after,
        source_script="scripts/xl_no_pca_table.py",
    )


if __name__ == "__main__":
    main()
