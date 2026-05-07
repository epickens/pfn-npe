"""Generate compact manuscript comparison tables from C2ST artifacts.

Outputs:
  - External SBI baseline table at n_train=10k.
  - Compact task-family win/median summary for the main text.

The local PFN-family rows come from
`pfn_testing/sbi/outputs/layer_ablation/c2st_decomp/`. Shipped SBIBM
baselines come from `sbibm.get_results()`, matching the source used by
`scripts/plot_budget_scaling.py`.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tex_table import write_tex_tabular  # noqa: E402


DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation")
PAPER_TABLE_DIR = Path("results/tables")
DEFAULT_BUDGET = 10000

NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)(?:_(?P<rest>.+))?$")
N_SUFFIX_RE = re.compile(r"_n(\d+)$")
SBIBM_BUDGET_MAP = {"10³": 1000, "10⁴": 10000, "10⁵": 100000}

TASK_LABELS = {
    "two_moons": "Two moons",
    "gaussian_mixture": "Gaussian mixture",
    "gaussian_linear": "Gaussian linear",
    "gaussian_linear_uniform": "Gaussian linear uniform",
    "bernoulli_glm": "Bernoulli GLM",
    "bernoulli_glm_raw": "Bernoulli GLM raw",
    "slcp": "SLCP",
    "slcp_distractors": "SLCP distractors",
    "sir": "SIR",
    "lotka_volterra": "Lotka-Volterra",
    "two_moons_distractors": "Two moons + distr.",
    "gaussian_mixture_distractors": "G. mixture + distr.",
    "bernoulli_glm_distractors": "Bern. GLM + distr.",
    "sir_distractors": "SIR + distr.",
    "ar1_ts_t50": "AR(1), T=50",
    "ou": "OU",
    "solar_dynamo": "Solar dynamo",
}

STANDARD_TASKS = [
    "two_moons",
    "gaussian_mixture",
    "gaussian_linear",
    "bernoulli_glm",
    "slcp",
    "sir",
    "lotka_volterra",
]

FAMILIES = {
    "SBIBM reference": [
        "two_moons",
        "gaussian_mixture",
        "slcp",
        "slcp_distractors",
        "gaussian_linear",
        "gaussian_linear_uniform",
        "bernoulli_glm",
        "bernoulli_glm_raw",
        "sir",
        "lotka_volterra",
    ],
    "Distractor variants": [
        "two_moons_distractors",
        "gaussian_mixture_distractors",
        "bernoulli_glm_distractors",
        "sir_distractors",
    ],
    "Time series": [
        "ar1_ts_t50",
        "ou",
        "solar_dynamo",
    ],
}

LOCAL_METHOD_LABELS = {
    "nsf": "PFN-NPE",
    "learned_summary_npe": "Learned summary + NSF",
    "npe_pfn": "NPE-PFN",
    "fmpe": "FMPE",
}


def parse_name(stem: str) -> tuple[str, int, str, int] | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    rest = m.group("rest") or ""
    n_match = N_SUFFIX_RE.search(rest)
    if n_match:
        budget = int(n_match.group(1))
        method = rest[: n_match.start()]
    else:
        budget = DEFAULT_BUDGET
        method = rest
    if not method:
        method = "nsf"
    return m.group("task"), int(m.group("seed")), method, budget


def load_local(metric: str, budget: int) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in sorted(DECOMP_DIR.glob("*.npz")):
        parsed = parse_name(path.stem)
        if not parsed:
            continue
        task, _seed, method, parsed_budget = parsed
        if parsed_budget != budget:
            continue
        try:
            data = np.load(path, allow_pickle=True)
            arr = data[metric]
        except (KeyError, OSError):
            continue
        values[(task, method)].append(float(np.mean(arr)))
    return values


def local_mean(values: dict[tuple[str, str], list[float]], task: str, method: str) -> float | None:
    vals = values.get((task, method))
    if not vals:
        return None
    return float(np.mean(vals))


def fmt(value: float | None, decimals: int, bold: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "---"
    body = f"{value:.{decimals}f}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return f"${body}$"


def load_sbibm_baselines(
    tasks: list[str],
    algorithms: list[str],
    budget: int,
) -> dict[tuple[str, str], float]:
    try:
        import sbibm
    except Exception as exc:  # pragma: no cover - depends on optional env
        print(f"[warn] sbibm unavailable; external baselines omitted: {exc}")
        return {}

    df = sbibm.get_results()
    df = df[df["task"].isin(tasks) & df["algorithm"].isin(algorithms)].copy()
    df["budget_int"] = df["num_simulations"].map(SBIBM_BUDGET_MAP)
    df = df[df["budget_int"] == budget]

    out: dict[tuple[str, str], float] = {}
    for (task, algorithm), sub in df.groupby(["task", "algorithm"]):
        out[(task, algorithm)] = float(sub["C2ST"].mean())
    return out


def write_external_table(args: argparse.Namespace) -> None:
    local = load_local(metric=args.metric, budget=args.budget)
    algorithms = ["NPE", "NLE", "NRE", "SNPE", "SNLE"]
    shipped = load_sbibm_baselines(STANDARD_TASKS, algorithms, args.budget)

    # FMPE is not a shipped SBIBM result; include it only if local cells exist.
    include_fmpe = any((task, "fmpe") in local for task in STANDARD_TASKS)
    columns = ["Task", "PFN-NPE", "NPE", "NLE", "NRE"]
    if include_fmpe:
        columns.append("FMPE")
    columns.extend(["SNPE", "SNLE"])

    rows: list[list[str]] = []
    for task in STANDARD_TASKS:
        entries: dict[str, float | None] = {
            "PFN-NPE": local_mean(local, task, "nsf"),
            "NPE": shipped.get((task, "NPE")),
            "NLE": shipped.get((task, "NLE")),
            "NRE": shipped.get((task, "NRE")),
            "SNPE": shipped.get((task, "SNPE")),
            "SNLE": shipped.get((task, "SNLE")),
        }
        if include_fmpe:
            entries["FMPE"] = local_mean(local, task, "fmpe")
        finite = [v for v in entries.values() if v is not None and np.isfinite(v)]
        best = min(finite) if finite else None
        row = [TASK_LABELS[task]]
        for col in columns[1:]:
            value = entries.get(col)
            row.append(fmt(value, args.external_decimals, bold=best is not None and value == best))
        rows.append(row)

    write_tex_tabular(
        out_path=args.external_tex_out,
        columns=columns,
        rows=rows,
        column_align="l" + "c" * (len(columns) - 1),
        source_script="scripts/paper_comparison_tables.py",
    )


def write_compact_family_table(args: argparse.Namespace) -> None:
    local = load_local(metric=args.metric, budget=args.budget)
    methods = ["nsf", "learned_summary_npe", "npe_pfn"]
    win_labels = {
        "nsf": "PFN-NPE best",
        "learned_summary_npe": "Learned summary best",
        "npe_pfn": "NPE-PFN best",
    }
    columns = [
        "Task family",
        r"\# tasks",
        win_labels["nsf"],
        win_labels["learned_summary_npe"],
        win_labels["npe_pfn"],
        "Median PFN-NPE",
        "Median best",
    ]
    rows: list[list[str]] = []
    for family, tasks in FAMILIES.items():
        pfn_vals: list[float] = []
        best_vals: list[float] = []
        wins = {m: 0 for m in methods}
        n_tasks = 0
        for task in tasks:
            method_vals = {
                method: local_mean(local, task, method)
                for method in methods
            }
            available = {
                method: value
                for method, value in method_vals.items()
                if value is not None and np.isfinite(value)
            }
            if not available:
                continue
            n_tasks += 1
            if method_vals["nsf"] is not None:
                pfn_vals.append(float(method_vals["nsf"]))
            best_method = min(available, key=available.get)
            wins[best_method] += 1
            best_vals.append(float(available[best_method]))

        rows.append([
            family,
            str(n_tasks),
            str(wins["nsf"]),
            str(wins["learned_summary_npe"]),
            str(wins["npe_pfn"]),
            fmt(float(np.median(pfn_vals)) if pfn_vals else None, args.compact_decimals),
            fmt(float(np.median(best_vals)) if best_vals else None, args.compact_decimals),
        ])

    write_tex_tabular(
        out_path=args.compact_tex_out,
        columns=columns,
        rows=rows,
        column_align="lrrrrcc",
        source_script="scripts/paper_comparison_tables.py",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="joint", choices=["joint", "marginal", "rank"])
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--external-decimals", type=int, default=2)
    ap.add_argument("--compact-decimals", type=int, default=3)
    ap.add_argument(
        "--external-tex-out",
        type=Path,
        default=PAPER_TABLE_DIR / "external_baselines_10k.tex",
    )
    ap.add_argument(
        "--compact-tex-out",
        type=Path,
        default=PAPER_TABLE_DIR / "compact_family_summary.tex",
    )
    args = ap.parse_args()

    if not DECOMP_DIR.exists():
        sys.exit(f"Missing C2ST decomposition directory: {DECOMP_DIR}")

    write_external_table(args)
    write_compact_family_table(args)


if __name__ == "__main__":
    main()
