"""Wall-clock scaling curves for NPE, NPE-PFN, and PFN-NPE.

Runs methods over simulation budgets on a low-dimensional task and a
higher-dimensional task, then writes a CSV plus a two-panel appendix figure.

Default figure:
  - left:  two_moons, d_theta=2
  - right: bernoulli_glm, d_theta=10

Example:
  uv run python scripts/wall_clock_scaling.py
  uv run python scripts/wall_clock_scaling.py --from-csv path/to/scaling.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/wall_clock"
FIG_DIR = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/figures"
FLOW_VS_QUANTILE = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile"

TIMING_RE = re.compile(r"^\[TIMING\] phase=(\w+) duration=([\d.]+)", re.MULTILINE)
DIM_THETA_RE = re.compile(r"dim_theta=(\d+)")

TASK_LABELS = {
    "two_moons": "Two moons",
    "bernoulli_glm": "Bernoulli GLM",
}
METHOD_LABELS = {
    "npe_pfn": "NPE-PFN",
    "pfn_npe_nsf": "PFN-NPE",
    "vanilla_npe": "NPE",
    "learned_summary_npe": "Learned-summary NPE",
}
METHOD_STYLES = {
    "npe_pfn": {"color": "#222222", "marker": "D"},
    "pfn_npe_nsf": {"color": "#0072B2", "marker": "o"},
    "vanilla_npe": {"color": "#009E73", "marker": "^"},
    "learned_summary_npe": {"color": "#CC79A7", "marker": "P"},
}

TASKS_DEFAULT = ["two_moons", "bernoulli_glm"]
BUDGETS_DEFAULT = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
SEEDS_DEFAULT = [1000, 1001, 1002]
METHODS_DEFAULT = ["vanilla_npe", "learned_summary_npe", "pfn_npe_nsf", "npe_pfn"]


def effective_context(
    budget: int,
    *,
    cap: int,
    mode: str,
    fixed_context: int,
) -> int:
    if mode == "fixed":
        return fixed_context
    return min(budget, cap)


def method_command(
    method: str,
    *,
    task: str,
    seed: int,
    budget: int,
    n_val: int,
    n_ref: int,
    n_flow_samples: int,
    context_size: int,
    context_cap: int,
    max_epochs: int,
) -> list[str]:
    common = [
        "--task", task,
        "--seed", str(seed),
        "--n-train", str(budget),
        "--n-val", str(n_val),
        "--n-ref", str(n_ref),
        "--n-flow-samples", str(n_flow_samples),
    ]
    if method == "npe_pfn":
        return [
            sys.executable, "scripts/npe_pfn_baseline.py",
            *common,
            "--filter-context-size", str(context_cap),
        ]
    if method == "pfn_npe_nsf":
        return [
            sys.executable, "scripts/train_and_sample_flow.py",
            "--flow-type", "nsf",
            *common,
            "--context-size", str(context_size),
            "--max-epochs", str(max_epochs),
        ]
    if method == "vanilla_npe":
        return [sys.executable, "scripts/vanilla_npe_baseline.py", *common]
    if method == "learned_summary_npe":
        return [
            sys.executable,
            "scripts/learned_summary_npe_baseline.py",
            *common,
            "--max-epochs",
            str(max_epochs),
        ]
    raise ValueError(f"Unknown method: {method}")


def cleanup_outputs(task: str, seed: int, budget: int, methods: list[str]) -> None:
    """Remove timing outputs so subprocesses do not hit their skip path."""
    suffixes: list[str] = []
    if "npe_pfn" in methods:
        suffixes.append("npe_pfn" if budget == 10_000 else f"npe_pfn_n{budget}")
    if "pfn_npe_nsf" in methods:
        suffixes.append("nsf" if budget == 10_000 else f"nsf_n{budget}")
    if "vanilla_npe" in methods:
        suffixes.append("vanilla_npe" if budget == 10_000 else f"vanilla_npe_n{budget}")
    if "learned_summary_npe" in methods:
        suffixes.append(
            "learned_summary_npe"
            if budget == 10_000
            else f"learned_summary_npe_n{budget}"
        )

    for suffix in suffixes:
        for path in FLOW_VS_QUANTILE.glob(f"{task}_s{seed}_{suffix}*.npz"):
            path.unlink(missing_ok=True)


def run_and_time(cmd: list[str]) -> dict[str, float | int | None] | None:
    print(f"  $ {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    total = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        sys.stderr.write(result.stderr[-2000:])
        sys.stderr.write("\n")
        return None

    phases = {m.group(1): float(m.group(2)) for m in TIMING_RE.finditer(result.stdout)}
    dim_match = DIM_THETA_RE.search(result.stdout)
    return {
        "train": phases.get("train"),
        "sample": phases.get("sample"),
        "total": total,
        "dim_theta": int(dim_match.group(1)) if dim_match else None,
    }


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task", "dim_theta", "budget", "seed", "method", "context_size",
        "train_s", "sample_s", "total_s",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


def aggregate(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[int, dict[str, list[float]]]]]:
    by_task: dict[str, dict[str, dict[int, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for row in rows:
        task = row["task"]
        method = row["method"]
        budget = int(row["budget"])
        for phase in ("train", "sample", "total"):
            value = row.get(f"{phase}_s", "")
            if value:
                by_task[task][method][budget][phase].append(float(value))
    return by_task


def plot_scaling(
    rows: list[dict[str, str]],
    *,
    tasks: list[str],
    methods: list[str],
    context_cap: int,
    out: Path,
    yscale: str,
    errorbar: str,
) -> None:
    import matplotlib.pyplot as plt

    by_task = aggregate(rows)
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 2.8), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks, strict=False):
        for method in methods:
            by_budget = by_task.get(task, {}).get(method, {})
            budgets = sorted(by_budget)
            if not budgets:
                continue
            means = [float(np.mean(by_budget[b]["total"])) for b in budgets]
            yerr = None
            if errorbar != "none":
                spreads = []
                for budget in budgets:
                    values = by_budget[budget]["total"]
                    if len(values) <= 1:
                        spreads.append(0.0)
                        continue
                    spread = float(np.std(values))
                    if errorbar == "sem":
                        spread /= float(np.sqrt(len(values)))
                    spreads.append(spread)
                yerr = spreads
            style = METHOD_STYLES.get(method, {})
            ax.errorbar(
                budgets,
                means,
                yerr=yerr,
                label=METHOD_LABELS.get(method, method),
                linewidth=2.2,
                markersize=4.5,
                capsize=2.5 if yerr is not None else 0,
                **style,
            )

        ax.axvline(context_cap, color="#777777", linestyle=":", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_title(TASK_LABELS.get(task, task.replace("_", " ")), fontsize=10)
        ax.set_xlabel("Training simulations")
        ax.grid(True, alpha=0.2)
        dim_values = sorted({
            row["dim_theta"] for row in rows
            if row["task"] == task and row.get("dim_theta")
        })
        if dim_values:
            ax.text(
                0.04, 0.92, f"d_theta={dim_values[0]}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )

    axes[0].set_ylabel("Total wall-clock time (s)")
    handles, _ = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}\nWrote {pdf_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS_DEFAULT)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument("--methods", nargs="+", default=METHODS_DEFAULT,
                    choices=list(METHOD_LABELS))
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--n-flow-samples", type=int, default=1000)
    ap.add_argument("--context-cap", type=int, default=10_000)
    ap.add_argument("--pfn-context-mode", choices=["capped", "fixed"], default="capped",
                    help="PFN-NPE context sweep. 'capped' uses min(n_train, cap); "
                         "'fixed' keeps --fixed-pfn-context.")
    ap.add_argument("--fixed-pfn-context", type=int, default=1000)
    ap.add_argument("--max-epochs", type=int, default=200,
                    help="PFN-NPE flow training epochs.")
    ap.add_argument("--csv-out", type=Path, default=OUT_DIR / "wall_clock_scaling.csv")
    ap.add_argument("--plot-out", type=Path, default=FIG_DIR / "wall_clock_scaling.png")
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="Skip running subprocesses and only plot an existing CSV.")
    ap.add_argument("--keep-outputs", action="store_true",
                    help="Keep generated flow_vs_quantile timing outputs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without running them.")
    ap.add_argument("--yscale", choices=["linear", "log"], default="log")
    ap.add_argument(
        "--errorbar",
        choices=["std", "sem", "none"],
        default="std",
        help="Error bars across timing seeds in the appendix figure.",
    )
    args = ap.parse_args()

    if args.from_csv is not None:
        rows = read_rows(args.from_csv)
        plot_scaling(
            rows,
            tasks=args.tasks,
            methods=args.methods,
            context_cap=args.context_cap,
            out=args.plot_out,
            yscale=args.yscale,
            errorbar=args.errorbar,
        )
        return

    rows: list[dict[str, str]] = []
    for task in args.tasks:
        for budget in args.budgets:
            context_size = effective_context(
                budget,
                cap=args.context_cap,
                mode=args.pfn_context_mode,
                fixed_context=args.fixed_pfn_context,
            )
            for seed in args.seeds:
                if not args.keep_outputs and not args.dry_run:
                    cleanup_outputs(task, seed, budget, args.methods)
                for method in args.methods:
                    print(f"\n[{task}, n={budget}, seed={seed}, method={method}]")
                    cmd = method_command(
                        method,
                        task=task,
                        seed=seed,
                        budget=budget,
                        n_val=args.n_val,
                        n_ref=args.n_ref,
                        n_flow_samples=args.n_flow_samples,
                        context_size=context_size,
                        context_cap=args.context_cap,
                        max_epochs=args.max_epochs,
                    )
                    if args.dry_run:
                        print("  $ " + " ".join(cmd))
                        continue
                    timing = run_and_time(cmd)
                    rows.append({
                        "task": task,
                        "dim_theta": "" if timing is None or timing["dim_theta"] is None
                                     else str(timing["dim_theta"]),
                        "budget": str(budget),
                        "seed": str(seed),
                        "method": method,
                        "context_size": str(context_size),
                        "train_s": "" if timing is None or timing["train"] is None
                                   else f"{timing['train']:.3f}",
                        "sample_s": "" if timing is None or timing["sample"] is None
                                    else f"{timing['sample']:.3f}",
                        "total_s": "" if timing is None or timing["total"] is None
                                   else f"{timing['total']:.3f}",
                    })

    if args.dry_run:
        return

    write_rows(args.csv_out, rows)
    plot_scaling(
        rows,
        tasks=args.tasks,
        methods=args.methods,
        context_cap=args.context_cap,
        out=args.plot_out,
        yscale=args.yscale,
        errorbar=args.errorbar,
    )

    if not args.keep_outputs:
        for task in args.tasks:
            for budget in args.budgets:
                for seed in args.seeds:
                    cleanup_outputs(task, seed, budget, args.methods)


if __name__ == "__main__":
    main()
