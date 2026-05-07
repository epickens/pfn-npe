"""Plot amortized wall-clock runtime together with posterior quality.

Reads the timing CSV from `scripts/wall_clock_amortization.py` plus its optional
quality CSV and writes a compact two-panel figure:
  1. total wall-clock time vs number of posterior queries
  2. joint C2ST by method / flow size
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
    "pfn_npe_base": "PFN-NPE\nbase",
    "pfn_npe_large": "PFN-NPE\nlarge",
    "pfn_npe_xl": "PFN-NPE\nXL",
}
LINE_LABELS = {
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
    "slcp": "SLCP",
    "two_moons": "Two moons",
    "bernoulli_glm": "Bernoulli GLM",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std()) if len(arr) > 1 else 0.0


def aggregate_time(
    rows: list[dict[str, str]],
    task: str,
) -> dict[str, dict[int, list[float]]]:
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["task"] != task or row["status"] != "ok" or not row["total_s"]:
            continue
        out[row["method"]][int(row["n_query"])].append(float(row["total_s"]))
    return out


def aggregate_quality(
    rows: list[dict[str, str]],
    task: str,
) -> dict[str, list[float]]:
    by_seed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["task"] != task or row["status"] != "ok" or not row["joint_c2st"]:
            continue
        by_seed[row["method"]][row["seed"]].append(float(row["joint_c2st"]))

    out: dict[str, list[float]] = {}
    for method, seed_values in by_seed.items():
        out[method] = [float(np.mean(values)) for values in seed_values.values()]
    return out


def plot(
    timing_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
    *,
    task: str,
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    by_time = aggregate_time(timing_rows, task)
    by_quality = aggregate_quality(quality_rows, task)
    dim_values = sorted(
        {
            row["dim_theta"]
            for row in timing_rows
            if row["task"] == task and row["dim_theta"]
        }
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    ax_time, ax_q = axes

    for method in METHODS:
        by_q = by_time.get(method, {})
        q_values = sorted(by_q)
        if not q_values:
            continue
        means, stds = zip(*(mean_std(by_q[q]) for q in q_values), strict=True)
        ax_time.errorbar(
            q_values,
            means,
            yerr=stds,
            label=LINE_LABELS[method],
            linewidth=2.2,
            markersize=4.5,
            capsize=2.5,
            **METHOD_STYLES[method],
        )

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Posterior queries")
    ax_time.set_ylabel("Total wall-clock time (s)")
    ax_time.grid(True, alpha=0.2)
    ax_time.set_title("A. Amortized runtime", fontsize=10)
    if dim_values:
        ax_time.text(
            0.04,
            0.92,
            f"d_theta={dim_values[0]}",
            transform=ax_time.transAxes,
            fontsize=8,
            va="top",
        )

    xs = np.arange(len(METHODS))
    quality_means = []
    quality_stds = []
    for method in METHODS:
        vals = by_quality.get(method, [])
        if vals:
            m, s = mean_std(vals)
        else:
            m, s = np.nan, 0.0
        quality_means.append(m)
        quality_stds.append(s)

    for i, method in enumerate(METHODS):
        color = METHOD_STYLES[method]["color"]
        ax_q.errorbar(
            [xs[i]],
            [quality_means[i]],
            yerr=[quality_stds[i]],
            marker=METHOD_STYLES[method]["marker"],
            color=color,
            markersize=6,
            capsize=3,
            linewidth=0,
        )
        seed_vals = by_quality.get(method, [])
        jitter = np.linspace(-0.07, 0.07, len(seed_vals)) if seed_vals else []
        ax_q.scatter(
            xs[i] + jitter,
            seed_vals,
            s=12,
            color=color,
            alpha=0.45,
            linewidths=0,
        )

    ax_q.set_xticks(xs)
    ax_q.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8)
    ax_q.set_ylabel("Joint C2ST")
    ax_q.set_title("B. Posterior quality", fontsize=10)
    ax_q.grid(True, axis="y", alpha=0.2)
    ax_q.set_ylim(0.75, 0.86)

    fig.suptitle(TASK_LABELS.get(task, task.replace("_", " ")), y=1.02, fontsize=12)
    handles, labels = ax_time.get_legend_handles_labels()
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
    ap.add_argument("--timing-csv", type=Path, required=True)
    ap.add_argument("--quality-csv", type=Path, required=True)
    ap.add_argument("--task", default="slcp")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "pfn_testing/sbi/outputs/layer_ablation/figures/wall_clock_amortization_slcp_quality.png"
        ),
    )
    args = ap.parse_args()

    plot(
        read_csv(args.timing_csv),
        read_csv(args.quality_csv),
        task=args.task,
        out=args.out,
    )


if __name__ == "__main__":
    main()
