"""Aggregate C2ST decomposition results across seeds into a paper-ready table.

Reads every npz in `pfn_testing/sbi/outputs/layer_ablation/c2st_decomp/`,
groups by (task, method), and reports mean ± std across seeds (each seed's
contribution is itself a mean over the n_ref reference observations).

Outputs:
  - long CSV at outputs/layer_ablation/c2st_summary.csv
    columns: task, method, metric, mean, std, n_seeds, n_obs
  - wide tables to stdout (and optionally LaTeX) for each metric

Usage:
  uv run python scripts/aggregate_c2st_table.py
  uv run python scripts/aggregate_c2st_table.py --tasks slcp two_moons --methods nsf our_ar copula copula_neural
  uv run python scripts/aggregate_c2st_table.py --budget 50000
  uv run python scripts/aggregate_c2st_table.py --budget 50000 --out-csv /tmp/c2st_50k.csv
  uv run python scripts/aggregate_c2st_table.py --latex
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
OUT_CSV = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_summary.csv")
DEFAULT_BUDGET = 10000

NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)(?:_(?P<method>.+))?$")
N_SUFFIX_RE = re.compile(r"_n(\d+)$")

METHODS_DEFAULT = [
    "nsf", "rank_dr", "learned_gibbs", "blocked_gibbs", "our_ar",
    "residual_flow", "copula", "copula_neural",
]
METHOD_LABELS = {
    "":                    "PFN-NPE",      # vanilla nsf, unsuffixed legacy
    "nsf":                 "PFN-NPE",
    "nsf_no_pca":          "PFN-NPE (no PCA)",
    "nsf_filter":          "NSF+filter",
    "rank_dr":             "Rank-DR",
    "learned_gibbs":       "Learned Gibbs",
    "blocked_gibbs":       "Blocked Gibbs",
    "our_ar":              "AR (ours)",
    "npe_pfn":             "NPE-PFN",
    "residual_flow":       "AR+flow",
    "copula":              "Cop-Gauss",
    "copula_neural":       "Cop-Neural",
    "copula_fully_neural": "Cop-FullN",
    "learned_summary_npe": "Learned summary + NSF",
}
METRIC_LABELS = {"joint": "Joint", "marginal": "Marg", "rank": "Rank"}


def parse_name(stem: str) -> tuple[str, int, str, int] | None:
    """Parse `{task}_s{seed}[_{method}][_n{budget}]`.

    Empty/unsuffixed method (legacy `{task}_s{seed}.npz`) is normalized to
    'nsf' so that pre-suffix-convention vanilla-NSF cells aggregate
    together with modern `_nsf.npz` cells. When both exist for the same
    (task, seed), the modern suffixed cell sorts later and wins via dict
    overwrite — preferred since it's the canonical record.
    """
    m = NAME_RE.match(stem)
    if not m:
        return None
    rest = m.group("method") or ""
    n_match = N_SUFFIX_RE.search(rest)
    if n_match:
        budget = int(n_match.group(1))
        method = rest[:n_match.start()]
    else:
        budget = DEFAULT_BUDGET
        method = rest
    if not method:
        method = "nsf"
    return m.group("task"), int(m.group("seed")), method, budget


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Restrict to these tasks (default: all on disk).")
    ap.add_argument("--methods", nargs="*", default=METHODS_DEFAULT,
                    help=f"Methods (suffix tags). Default: {METHODS_DEFAULT}")
    ap.add_argument("--metric", default="joint",
                    choices=["joint", "marginal", "rank", "all"],
                    help="Metric for the wide stdout table.")
    ap.add_argument("--latex", action="store_true",
                    help="Also emit a LaTeX tabular block to stdout.")
    ap.add_argument("--tex-out", type=Path, default=None,
                    help="Write a self-contained LaTeX tabular fragment to "
                         "this path (one fragment for the chosen metric; "
                         "use 'joint' / 'marginal' / 'rank', not 'all'). "
                         "Suitable for `\\input{}` from the writeup.")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="Training-simulation budget to aggregate. Files "
                         "without an `_n{budget}` suffix are treated as "
                         f"{DEFAULT_BUDGET}.")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Long CSV output path. Defaults to c2st_summary.csv "
                         f"for budget={DEFAULT_BUDGET}, otherwise "
                         "c2st_summary_n{budget}.csv.")
    ap.add_argument("--decimals", type=int, default=3)
    args = ap.parse_args()

    if not DECOMP_DIR.exists():
        sys.exit(f"Missing decomp dir: {DECOMP_DIR}")

    # ── Load and group ───────────────────────────────────────────────────
    # data[(task, method)][seed] = {joint, marginal, rank}  (each: array of n_ref)
    data: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for p in sorted(DECOMP_DIR.glob("*.npz")):
        parsed = parse_name(p.stem)
        if not parsed:
            continue
        task, seed, method, budget = parsed
        if budget != args.budget:
            continue
        if args.tasks is not None and task not in args.tasks:
            continue
        if method not in args.methods:
            continue
        try:
            d = np.load(p, allow_pickle=True)
        except OSError:
            continue
        data[(task, method)][seed] = {
            "joint": d["joint"], "marginal": d["marginal"], "rank": d["rank"],
        }

    if not data:
        sys.exit("No matching c2st_decomp files found.")

    tasks = sorted({t for t, _ in data})
    methods_present = [m for m in args.methods if any(
        (t, m) in data for t in tasks
    )]

    # ── Aggregate ────────────────────────────────────────────────────────
    # For each (task, method, metric) compute:
    #   per-seed-mean values, then mean ± std across seeds
    rows: list[dict] = []
    summary: dict[tuple[str, str, str], tuple[float, float, int, int]] = {}
    for (task, method), seed_dict in data.items():
        for metric in ("joint", "marginal", "rank"):
            seed_means = np.array([
                np.mean(d[metric]) for d in seed_dict.values()
            ])
            n_seeds = len(seed_means)
            n_obs = sum(len(d[metric]) for d in seed_dict.values())
            mean = float(seed_means.mean())
            std = float(seed_means.std(ddof=0)) if n_seeds > 1 else 0.0
            summary[(task, method, metric)] = (mean, std, n_seeds, n_obs)
            rows.append({
                "task": task, "method": method, "metric": metric,
                "mean": mean, "std": std,
                "n_seeds": n_seeds, "n_obs": n_obs,
            })

    # ── Write long CSV ───────────────────────────────────────────────────
    out_csv = args.out_csv
    if out_csv is None:
        out_csv = (
            OUT_CSV if args.budget == DEFAULT_BUDGET
            else OUT_CSV.with_name(f"c2st_summary_n{args.budget}.csv")
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("task,method,metric,mean,std,n_seeds,n_obs\n")
        for r in rows:
            f.write(
                f"{r['task']},{r['method']},{r['metric']},"
                f"{r['mean']:.6f},{r['std']:.6f},"
                f"{r['n_seeds']},{r['n_obs']}\n"
            )
    print(f"Wrote {out_csv} ({len(rows)} rows, budget={args.budget})")
    print()

    # ── Print wide table(s) ──────────────────────────────────────────────
    metrics = ["joint", "marginal", "rank"] if args.metric == "all" else [args.metric]
    fmt = f".{args.decimals}f"

    for metric in metrics:
        print(f"=== {METRIC_LABELS[metric]} C2ST  (mean ± std across seeds) ===")
        header = f"{'task':<32} " + " ".join(
            f"{METHOD_LABELS.get(m, m):>16}" for m in methods_present
        )
        print(header)
        for task in tasks:
            cells = []
            for m in methods_present:
                if (task, m, metric) not in summary:
                    cells.append(f"{'-':>16}")
                else:
                    mean, std, ns, _ = summary[(task, m, metric)]
                    if ns >= 2:
                        cells.append(f"{mean:{fmt}}±{std:.2f} (n={ns})".rjust(16))
                    else:
                        cells.append(f"{mean:{fmt}} (n=1)".rjust(16))
            print(f"{task:<32} " + " ".join(cells))
        print()

    # ── Optional LaTeX file fragment ─────────────────────────────────────
    if args.tex_out is not None:
        if args.metric == "all":
            sys.exit("--tex-out requires a single metric (not 'all').")
        metric = args.metric
        columns = ["Task"] + [
            METHOD_LABELS.get(m, m) for m in methods_present
        ]
        out_rows: list[list[str]] = []
        for task in tasks:
            row_means = [
                summary.get((task, m, metric), (np.inf, 0, 0, 0))[0]
                for m in methods_present
            ]
            best_idx = int(np.argmin(row_means)) if row_means else -1
            cells = [task.replace("_", r"\_")]
            for j, m in enumerate(methods_present):
                if (task, m, metric) not in summary:
                    cells.append("---")
                    continue
                mean, std, ns, _ = summary[(task, m, metric)]
                # Bold the mean only; \textbf inside $...$ would break \pm.
                mean_str = (rf"\mathbf{{{mean:{fmt}}}}"
                            if j == best_idx and ns >= 1
                            else f"{mean:{fmt}}")
                txt = (f"{mean_str}\\,\\pm\\,{std:.2f}"
                       if ns >= 2 else mean_str)
                cells.append(f"${txt}$")
            out_rows.append(cells)
        write_tex_tabular(
            out_path=args.tex_out,
            columns=columns,
            rows=out_rows,
            column_align="l" + "c" * len(methods_present),
            source_script="scripts/aggregate_c2st_table.py",
        )

    # ── Optional LaTeX to stdout ─────────────────────────────────────────
    if args.latex:
        for metric in metrics:
            print(f"% LaTeX — {METRIC_LABELS[metric]} C2ST")
            cols = "l" + "c" * len(methods_present)
            print(r"\begin{tabular}{" + cols + r"}")
            print(r"\toprule")
            head = "Task & " + " & ".join(
                METHOD_LABELS.get(m, m) for m in methods_present
            ) + r" \\"
            print(head)
            print(r"\midrule")
            for task in tasks:
                cells = []
                # Find best mean per row (lowest = best for C2ST)
                row_means = [
                    summary.get((task, m, metric), (np.inf, 0, 0, 0))[0]
                    for m in methods_present
                ]
                best_idx = int(np.argmin(row_means)) if row_means else -1
                for j, m in enumerate(methods_present):
                    if (task, m, metric) not in summary:
                        cells.append("-")
                        continue
                    mean, std, ns, _ = summary[(task, m, metric)]
                    txt = (f"{mean:{fmt}}\\,\\pm\\,{std:.2f}"
                           if ns >= 2 else f"{mean:{fmt}}")
                    if j == best_idx and ns >= 1:
                        txt = r"\textbf{" + txt + "}"
                    cells.append(f"${txt}$")
                tname = task.replace("_", r"\_")
                print(f"{tname} & " + " & ".join(cells) + r" \\")
            print(r"\bottomrule")
            print(r"\end{tabular}")
            print()


if __name__ == "__main__":
    main()
