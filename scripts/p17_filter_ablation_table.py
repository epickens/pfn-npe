"""Generate the P17 PFN-NPE filter diagnostic appendix table.

The table compares high-budget PFN-NPE/NSF with and without the AR-style
standardized-Euclidean top-k filter. The delta column is paired by seed:

    Delta = C2ST(PFN-NPE + filter) - C2ST(PFN-NPE)

Lower C2ST is better, so positive deltas mean the filter did not provide an
unexpected performance boost.

Usage:
    uv run python scripts/p17_filter_ablation_table.py
    uv run python scripts/p17_filter_ablation_table.py \\
        --tex-out pfn_testing/sbi/outputs/layer_ablation/tables/p17_filter_ablation_joint.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tex_table import write_tex_tabular  # noqa: E402

DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
TASKS_DEFAULT = ["ar1_ts_t50", "bernoulli_glm", "slcp"]
SEEDS_DEFAULT = [42, 7, 123]
METRICS = ["joint", "marginal", "rank"]
METHODS = [
    ("nsf", "PFN-NPE"),
    ("nsf_filter", "PFN-NPE+filter"),
    ("our_ar", "AR"),
    ("npe_pfn", "NPE-PFN"),
]


def tex_escape_task(task: str) -> str:
    return r"\texttt{" + task.replace("_", r"\_") + "}"


def load_seed_values(
    *,
    task: str,
    method: str,
    metric: str,
    budget: int,
    seeds: list[int],
) -> dict[int, float]:
    values: dict[int, float] = {}
    for seed in seeds:
        path = DECOMP_DIR / f"{task}_s{seed}_{method}_n{budget}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        values[seed] = float(np.mean(data[metric]))
    return values


def fmt_pm(values: np.ndarray, *, decimals: int, bold: bool = False) -> str:
    mean = float(values.mean())
    std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
    mean_str = f"{mean:.{decimals}f}"
    if bold:
        mean_str = rf"\mathbf{{{mean_str}}}"
    return rf"${mean_str}\,\pm\,{std:.{decimals}f}$"


def fmt_delta(values: np.ndarray, *, decimals: int) -> str:
    mean = float(values.mean())
    std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
    return rf"${mean:+.{decimals}f}\,\pm\,{std:.{decimals}f}$"


def build_rows(
    *,
    tasks: list[str],
    seeds: list[int],
    metric: str,
    budget: int,
    decimals: int,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for task in tasks:
        values_by_method = {
            method: load_seed_values(
                task=task,
                method=method,
                metric=metric,
                budget=budget,
                seeds=seeds,
            )
            for method, _ in METHODS
        }
        common = set(seeds)
        for method, _ in METHODS:
            common &= set(values_by_method[method])
        if not common:
            raise SystemExit(
                f"No complete {metric} cells for task={task}, budget={budget}"
            )
        paired_seeds = sorted(common)
        arrays = {
            method: np.array([values_by_method[method][s] for s in paired_seeds])
            for method, _ in METHODS
        }
        means = {method: float(vals.mean()) for method, vals in arrays.items()}
        best_method = min(means, key=means.get)
        delta = arrays["nsf_filter"] - arrays["nsf"]

        row = [
            tex_escape_task(task),
            fmt_pm(arrays["nsf"], decimals=decimals, bold=best_method == "nsf"),
            fmt_pm(
                arrays["nsf_filter"],
                decimals=decimals,
                bold=best_method == "nsf_filter",
            ),
            fmt_delta(delta, decimals=decimals),
            fmt_pm(arrays["our_ar"], decimals=decimals, bold=best_method == "our_ar"),
            fmt_pm(
                arrays["npe_pfn"],
                decimals=decimals,
                bold=best_method == "npe_pfn",
            ),
        ]
        rows.append(row)
    return rows


def print_plain(rows: list[list[str]]) -> None:
    header = [
        "task",
        "PFN-NPE",
        "PFN-NPE+filter",
        "delta",
        "AR",
        "NPE-PFN",
    ]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for row in rows:
        plain = [
            cell.replace(r"\texttt{", "").replace("}", "").replace(r"\_", "_")
            for cell in row
        ]
        plain = [cell.replace("$", "").replace(r"\,", "").replace(r"\pm", " +/- ") for cell in plain]
        plain = [cell.replace(r"\mathbf{", "").replace("}", "") for cell in plain]
        print(" | ".join(plain))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    ap.add_argument("--metric", choices=METRICS, default="joint")
    ap.add_argument("--budget", type=int, default=100000)
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--tex-out", type=Path, default=None,
                    help="Write LaTeX tabular fragment to this path.")
    args = ap.parse_args()

    if not DECOMP_DIR.exists():
        raise SystemExit(f"Missing decomp dir: {DECOMP_DIR}")

    rows = build_rows(
        tasks=args.tasks,
        seeds=args.seeds,
        metric=args.metric,
        budget=args.budget,
        decimals=args.decimals,
    )
    print_plain(rows)

    if args.tex_out is not None:
        write_tex_tabular(
            out_path=args.tex_out,
            columns=[
                "Task",
                "PFN-NPE",
                "PFN-NPE+filter",
                r"$\Delta$",
                "AR",
                "NPE-PFN",
            ],
            rows=rows,
            column_align="lccccc",
            source_script="scripts/p17_filter_ablation_table.py",
        )


if __name__ == "__main__":
    main()
