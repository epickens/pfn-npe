"""Plot runtime-only amortization curves across tasks.

Reads one or more timing CSVs from `scripts/wall_clock_amortization.py` and
writes a multi-panel figure with NPE-PFN and PFN-NPE flow-size curves.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ["npe_pfn", "pfn_npe_base", "pfn_npe_large", "pfn_npe_xl"]
METHOD_LABELS = {
    "npe_pfn": "NPE-PFN",
    "pfn_npe_base": "PFN-NPE base",
    "pfn_npe_large": "PFN-NPE large",
    "pfn_npe_xl": "PFN-NPE XL",
}
METHOD_STYLES = {
    "npe_pfn": {"color": "#222222", "marker": "D"},
    "pfn_npe_base": {"color": "#0072B2", "marker": "o"},
    "pfn_npe_large": {"color": "#009E73", "marker": "^"},
    "pfn_npe_xl": {"color": "#CC79A7", "marker": "s"},
}
TASK_LABELS = {
    "two_moons": "Two moons",
    "slcp": "SLCP",
    "bernoulli_glm": "Bernoulli GLM",
}


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std()) if len(arr) > 1 else 0.0


def aggregate(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[int, list[float]]]]:
    out: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        if row["status"] != "ok" or not row["total_s"]:
            continue
        out[row["task"]][row["method"]][int(row["n_query"])].append(
            float(row["total_s"])
        )
    return out


def plot(
    rows: list[dict[str, str]],
    *,
    tasks: list[str],
    out: Path,
    errorbar: str,
) -> None:
    import matplotlib.pyplot as plt

    by_task = aggregate(rows)
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(3.25 * len(tasks), 2.85), sharey=True
    )
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks, strict=False):
        for method in METHODS:
            by_q = by_task.get(task, {}).get(method, {})
            q_values = sorted(by_q)
            if not q_values:
                continue
            means = []
            spreads = []
            for q in q_values:
                mean, std = mean_std(by_q[q])
                means.append(mean)
                spreads.append(std)
            yerr = None
            if errorbar == "std":
                yerr = spreads
            elif errorbar == "sem":
                yerr = [
                    s / float(np.sqrt(len(by_q[q]))) if len(by_q[q]) > 1 else 0.0
                    for s, q in zip(spreads, q_values, strict=True)
                ]

            ax.errorbar(
                q_values,
                means,
                yerr=yerr,
                label=METHOD_LABELS[method],
                linewidth=2.2,
                markersize=4.5,
                capsize=2.5 if yerr is not None else 0,
                **METHOD_STYLES[method],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TASK_LABELS.get(task, task.replace("_", " ")), fontsize=10)
        ax.set_xlabel("Posterior queries")
        ax.grid(True, alpha=0.2)
        dim_values = sorted(
            {
                row["dim_theta"]
                for row in rows
                if row["task"] == task and row["dim_theta"]
            }
        )
        if dim_values:
            ax.text(
                0.05,
                0.92,
                f"d_theta={dim_values[0]}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )

    axes[0].set_ylabel("Total wall-clock time (s)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=8,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}\nWrote {pdf_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-csv", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--tasks", nargs="+", default=["two_moons", "slcp", "bernoulli_glm"]
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "pfn_testing/sbi/outputs/layer_ablation/figures/"
            "wall_clock_amortization_three_panel.png"
        ),
    )
    ap.add_argument("--errorbar", choices=["std", "sem", "none"], default="std")
    args = ap.parse_args()

    plot(
        read_rows(args.timing_csv),
        tasks=args.tasks,
        out=args.out,
        errorbar=args.errorbar,
    )


if __name__ == "__main__":
    main()
