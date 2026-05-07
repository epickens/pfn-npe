"""Generate paired TabPFN v2 vs v2.5 C2ST comparison tables.

Reads `c2st_decomp` cells for PFN-NPE/NSF with the default TabPFN v2 encoder
(`nsf`) and the TabPFN v2.5 encoder (`nsf_mv25`). For each task, aggregation is
restricted to seeds that exist for both versions, so the reported delta is a
paired comparison.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tex_table import write_tex_tabular  # noqa: E402

DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
TASKS_DEFAULT = [
    "slcp",
    "slcp_distractors",
    "bernoulli_glm_distractors",
    "two_moons_distractors",
    "ou",
]
METRICS_DEFAULT = ["joint", "marginal", "rank"]
METRIC_LABELS = {"joint": "joint", "marginal": "marginal", "rank": "rank"}
NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)(?:_(?P<method>.+))?$")


def parse_name(stem: str) -> tuple[str, int, str, bool] | None:
    """Parse a c2st_decomp stem and keep only v2/v2.5 NSF cells."""
    m = NAME_RE.match(stem)
    if not m:
        return None
    method = m.group("method") or ""
    if method == "":
        return m.group("task"), int(m.group("seed")), "nsf", False
    if method in {"nsf", "nsf_mv25"}:
        return m.group("task"), int(m.group("seed")), method, True
    return None


def tex_escape_task(task: str) -> str:
    return r"\texttt{" + task.replace("_", r"\_") + "}"


def fmt_pm(mean: float, std: float, *, bold: bool = False) -> str:
    mean_str = f"{mean:.3f}"
    if bold:
        mean_str = rf"\mathbf{{{mean_str}}}"
    return rf"${mean_str}\,\pm\,{std:.2f}$"


def fmt_delta(mean: float, std: float) -> str:
    return rf"${mean:+.3f}\,\pm\,{std:.2f}$"


def load_cells(tasks: list[str]) -> dict[tuple[str, str], dict[int, dict[str, np.ndarray]]]:
    """Load data[(task, method)][seed][metric].

    Legacy unsuffixed `{task}_s{seed}.npz` cells are treated as `nsf`, but
    explicit `{task}_s{seed}_nsf.npz` cells win when both are present.
    """
    data: dict[tuple[str, str], dict[int, dict[str, np.ndarray]]] = {}
    priority: dict[tuple[str, str, int], int] = {}
    for path in sorted(DECOMP_DIR.glob("*.npz")):
        parsed = parse_name(path.stem)
        if parsed is None:
            continue
        task, seed, method, explicit = parsed
        if task not in tasks:
            continue
        key = (task, method, seed)
        new_priority = 1 if explicit else 0
        if priority.get(key, -1) > new_priority:
            continue
        try:
            loaded = np.load(path, allow_pickle=True)
        except OSError:
            continue
        data.setdefault((task, method), {})[seed] = {
            "joint": np.asarray(loaded["joint"]),
            "marginal": np.asarray(loaded["marginal"]),
            "rank": np.asarray(loaded["rank"]),
        }
        priority[key] = new_priority
    return data


def build_rows(
    data: dict[tuple[str, str], dict[int, dict[str, np.ndarray]]],
    tasks: list[str],
    metric: str,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for task in tasks:
        v2 = data.get((task, "nsf"), {})
        v25 = data.get((task, "nsf_mv25"), {})
        seeds = sorted(set(v2) & set(v25))
        if not seeds:
            continue
        v2_seed_means = np.array([np.mean(v2[s][metric]) for s in seeds])
        v25_seed_means = np.array([np.mean(v25[s][metric]) for s in seeds])
        deltas = v25_seed_means - v2_seed_means

        v2_mean = float(v2_seed_means.mean())
        v25_mean = float(v25_seed_means.mean())
        best_is_v25 = v25_mean < v2_mean
        rows.append([
            tex_escape_task(task),
            str(len(seeds)),
            fmt_pm(
                v2_mean,
                float(v2_seed_means.std(ddof=0)) if len(seeds) > 1 else 0.0,
                bold=not best_is_v25,
            ),
            fmt_pm(
                v25_mean,
                float(v25_seed_means.std(ddof=0)) if len(seeds) > 1 else 0.0,
                bold=best_is_v25,
            ),
            fmt_delta(
                float(deltas.mean()),
                float(deltas.std(ddof=0)) if len(seeds) > 1 else 0.0,
            ),
        ])
    if not rows:
        raise SystemExit(f"No matched v2/v2.5 cells found for metric={metric}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    ap.add_argument("--metrics", nargs="+", default=METRICS_DEFAULT,
                    choices=METRICS_DEFAULT)
    ap.add_argument("--tex-dir", type=Path, required=True)
    ap.add_argument("--out-prefix", default="tabpfn_v25_comparison")
    args = ap.parse_args()

    if not DECOMP_DIR.exists():
        raise SystemExit(f"Missing decomp dir: {DECOMP_DIR}")

    data = load_cells(args.tasks)
    for metric in args.metrics:
        rows = build_rows(data, args.tasks, metric)
        out_path = args.tex_dir / f"{args.out_prefix}_{METRIC_LABELS[metric]}.tex"
        write_tex_tabular(
            out_path=out_path,
            columns=[
                "Task",
                "Seeds",
                "TabPFN v2",
                "TabPFN v2.5",
                r"$\Delta$",
            ],
            rows=rows,
            column_align="lcccc",
            source_script="scripts/tabpfn_v25_comparison_table.py",
        )


if __name__ == "__main__":
    main()
