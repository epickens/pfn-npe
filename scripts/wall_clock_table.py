"""Wall-clock comparison: NPE-PFN vs PFN-NPE training + inference time.

Times each method via subprocess on a single anchor task (default `slcp`)
across multiple timing-only seeds (1000+) that don't conflict with the
canonical (42, 7, 123) data on disk. Parses `[TIMING] phase=train|sample`
lines printed by the underlying scripts so the train / sample / total
breakdown lands in the appendix table.

Writes:
  - `pfn_testing/sbi/outputs/layer_ablation/wall_clock/wall_clock_{task}.csv`
  - prints a LaTeX-formattable summary block.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tex_table import write_tex_tabular  # noqa: E402

TIMING_RE = re.compile(r"\[TIMING\] phase=(\w+) duration=([\d.]+)")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/wall_clock"
FLOW_VS_QUANTILE = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile"

TASKS_DEFAULT = ["slcp", "two_moons_distractors", "gaussian_linear"]
SEEDS_DEFAULT = [1000, 1001, 1002]

# Prefer the project virtualenv Python when it exists.
VENV_PYTHON = REPO_ROOT / ".venv/bin/python"

# PyMC HMC supports only tasks with tractable analytic likelihoods.
PYMC_HMC_TASKS = {
    "slcp", "slcp_distractors",
    "gaussian_linear", "gaussian_linear_uniform",
    "gaussian_mixture", "gaussian_mixture_distractors",
}

METHODS = [
    {
        "name": "NPE-PFN",
        "label": "npe_pfn",
        "cmd": [str(VENV_PYTHON), "scripts/npe_pfn_baseline.py"],
        "applies_to": None,  # all tasks
    },
    {
        "name": "PFN-NPE (NSF)",
        "label": "pfn_npe_nsf",
        "cmd": [str(VENV_PYTHON), "scripts/train_and_sample_flow.py",
                "--flow-type", "nsf"],
        "applies_to": None,
    },
    {
        "name": "Learned summary + NSF",
        "label": "learned_summary_npe",
        "cmd": [str(VENV_PYTHON), "scripts/learned_summary_npe_baseline.py"],
        "applies_to": None,
    },
    {
        "name": "Vanilla NPE",
        "label": "vanilla_npe",
        "cmd": [str(VENV_PYTHON), "scripts/vanilla_npe_baseline.py"],
        "applies_to": None,
    },
    {
        "name": "PyMC HMC",
        "label": "pymc_hmc",
        "cmd": [str(VENV_PYTHON), "scripts/pymc_hmc_baseline.py"],
        "applies_to": PYMC_HMC_TASKS,
    },
]


def cleanup_outputs(task: str, seed: int) -> None:
    """Remove any output files at this (task, seed) so methods don't skip."""
    for p in FLOW_VS_QUANTILE.glob(f"{task}_s{seed}*.npz"):
        p.unlink(missing_ok=True)


def run_and_time(cmd: list[str], task: str, seed: int) -> dict | None:
    """Run command, return {'total': X, 'train': Y, 'sample': Z} (or None on failure)."""
    full_cmd = cmd + ["--task", task, "--seed", str(seed)]
    print(f"  $ {' '.join(full_cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(
        full_cmd, capture_output=True, text=True, cwd=REPO_ROOT,
    )
    total = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}):")
        sys.stderr.write(result.stderr[-1500:])
        sys.stderr.write("\n")
        return None
    phases = {m.group(1): float(m.group(2)) for m in TIMING_RE.finditer(result.stdout)}
    return {
        "total": total,
        "train": phases.get("train"),
        "sample": phases.get("sample"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Method labels to run (default: all). "
                         f"Choices: {[m['label'] for m in METHODS]}.")
    ap.add_argument("--keep-outputs", action="store_true",
                    help="Don't delete output files after timing")
    ap.add_argument("--tex-out", type=Path, default=None,
                    help="Optional path for a LaTeX tabular fragment. Writes "
                         "a self-contained `\\begin{tabular}...\\end{tabular}` "
                         "(no `\\caption` / `\\label`) suitable for `\\input{}`.")
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="Skip timing runs and regenerate the LaTeX "
                         "fragment from an existing wall_clock.csv.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    methods_to_run = METHODS
    if args.methods is not None:
        methods_to_run = [m for m in METHODS if m["label"] in args.methods]

    # Regenerate the LaTeX fragment from an existing timing CSV.
    if args.from_csv is not None:
        rows = []
        with args.from_csv.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
        if not rows:
            raise SystemExit(f"No rows in {args.from_csv}")
        if args.tex_out is not None:
            _write_tex_fragment(rows, methods_to_run, args.tasks, args.tex_out)
        _print_summary(rows, methods_to_run, args.tasks, args.seeds)
        return

    rows = []
    for task in args.tasks:
        for seed in args.seeds:
            cleanup_outputs(task, seed)  # Force methods to actually run.
            for method in methods_to_run:
                applies = method["applies_to"]
                if applies is not None and task not in applies:
                    print(f"\n[{method['name']}, task={task}, seed={seed}] "
                          f"-- skip (not in applicable task set)")
                    rows.append({
                        "method": method["name"], "label": method["label"],
                        "task": task, "seed": seed,
                        "train_s": "", "sample_s": "", "total_s": "",
                    })
                    continue
                print(f"\n[{method['name']}, task={task}, seed={seed}]")
                timing = run_and_time(method["cmd"], task, seed)
                rows.append({
                    "method": method["name"], "label": method["label"],
                    "task": task, "seed": seed,
                    "train_s": "" if timing is None or timing["train"] is None
                               else f"{timing['train']:.2f}",
                    "sample_s": "" if timing is None or timing["sample"] is None
                                else f"{timing['sample']:.2f}",
                    "total_s": "" if timing is None
                               else f"{timing['total']:.2f}",
                })

    csv_path = OUT_DIR / "wall_clock.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    if not args.keep_outputs:
        for task in args.tasks:
            for seed in args.seeds:
                cleanup_outputs(task, seed)
        print(f"Cleaned up timing outputs for tasks {args.tasks} seeds {args.seeds}")

    if args.tex_out is not None:
        _write_tex_fragment(rows, methods_to_run, args.tasks, args.tex_out)

    _print_summary(rows, methods_to_run, args.tasks, args.seeds)


def _aggregate(rows: list[dict], methods_to_run: list[dict], tasks: list[str]):
    """Returns dict[(task, method)] -> {'train': [...], 'sample': [...], 'total': [...]}."""
    by_cell: dict[tuple[str, str], dict[str, list[float]]] = {}
    method_names = [m["name"] for m in methods_to_run]
    for r in rows:
        if r["task"] not in tasks or r["method"] not in method_names:
            continue
        d = by_cell.setdefault(
            (r["task"], r["method"]),
            {"train": [], "sample": [], "total": []},
        )
        if not r.get("train_s") and r.get("sample_s") and r.get("total_s"):
            # Some subprocesses print the train timing marker on the same line
            # as library progress output. Older CSVs may therefore lack
            # train_s; use the residual so the table still partitions total
            # runtime into setup/training and sampling cost.
            try:
                r = dict(r)
                r["train_s"] = str(float(r["total_s"]) - float(r["sample_s"]))
            except ValueError:
                pass
        for k in ("train", "sample", "total"):
            v = r[f"{k}_s"]
            if v:
                d[k].append(float(v))
    return by_cell


def _fmt_cell(times: list[float]) -> str:
    if not times:
        return "—"
    m = float(np.mean(times))
    s = float(np.std(times)) if len(times) > 1 else 0.0
    return f"${m:.0f} \\pm {s:.0f}$"


def _write_tex_fragment(rows, methods_to_run, tasks, tex_out: Path):
    """Emit a multi-task wall-clock table fragment."""
    by_cell = _aggregate(rows, methods_to_run, tasks)
    columns = ["Task", "Method", "Train (s)", "Sample (s)", "Total (s)"]
    out_rows: list[list[str]] = []
    midrule_positions: list[int] = []
    pos = 0
    for ti, task in enumerate(tasks):
        for mi, method in enumerate(methods_to_run):
            d = by_cell.get((task, method["name"]))
            if d is None:
                continue
            label = task.replace("_", r"\_") if mi == 0 else ""
            out_rows.append([
                rf"\texttt{{{label}}}" if label else "",
                method["name"],
                _fmt_cell(d["train"]),
                _fmt_cell(d["sample"]),
                _fmt_cell(d["total"]),
            ])
            pos += 1
        if ti != len(tasks) - 1:
            midrule_positions.append(pos - 1)
    write_tex_tabular(
        out_path=tex_out,
        columns=columns,
        rows=out_rows,
        column_align="llrrr",
        midrules_after=midrule_positions,
        source_script="scripts/wall_clock_table.py",
    )


def _print_summary(rows, methods_to_run, tasks, seeds):
    by_cell = _aggregate(rows, methods_to_run, tasks)

    def fmt(times):
        if not times:
            return "—"
        m = float(np.mean(times))
        s = float(np.std(times)) if len(times) > 1 else 0.0
        return f"{m:.1f} ± {s:.1f}"

    for task in tasks:
        print(f"\n=== Wall-clock on {task} (n={len(seeds)} seeds) ===")
        print(f"{'method':<20} {'train (s)':>15} {'sample (s)':>15} "
              f"{'total (s)':>15}")
        for method in methods_to_run:
            d = by_cell.get((task, method["name"]))
            if d is None:
                continue
            print(f"{method['name']:<20} {fmt(d['train']):>15} "
                  f"{fmt(d['sample']):>15} {fmt(d['total']):>15}")


if __name__ == "__main__":
    main()
