"""Generate budget sweep plots with SBIBM baselines overlaid.

Loads our TabPFN + raw feature results from the outputs directory,
pulls SBIBM precomputed baselines (NPE, SNPE, SNLE), and optionally
includes fresh sbi baseline results.

Usage:
    uv run python -m pfn_testing.sbi.generate_plots
    uv run python -m pfn_testing.sbi.generate_plots --baselines NPE SNPE SNLE NLE
    uv run python -m pfn_testing.sbi.generate_plots --sbi-baselines npe nle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sbibm

from pfn_testing.sbi.plotting import plot_budget_curves, plot_budget_gap

OUTPUTS_DIR = Path(__file__).parent / "outputs"

TASKS = [
    "two_moons",
    "slcp",
    "slcp_distractors",
    "gaussian_mixture",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "lotka_volterra",
    "sir",
]

BUDGETS = [250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]

# SBIBM budget string -> int mapping
SBIBM_BUDGET_MAP = {"10³": 1000, "10⁴": 10000, "10⁵": 100000}


def load_our_results() -> dict[str, dict[int, dict]]:
    """Load TabPFN + raw results from outputs directory.

    Returns:
        {task_name: {budget: {"tabpfn": array(10,), "raw": array(10,)}}}
    """
    results = {}
    for task in TASKS:
        results[task] = {}
        for n in BUDGETS:
            npz_path = (
                OUTPUTS_DIR
                / task
                / f"n{n}_per_dim_regressor_pca_64"
                / "results"
                / "results.npz"
            )
            if not npz_path.exists():
                continue
            d = np.load(npz_path, allow_pickle=True)
            results[task][n] = {
                "tabpfn": d["c2st_tabpfn"],
                "raw": d["c2st_raw"],
            }
    return results


def load_sbibm_baselines(
    algorithms: list[str],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Load SBIBM precomputed baseline results.

    Returns:
        {task_name: {budget_int: {"NPE": array, "SNPE": array, ...}}}
    """
    df = sbibm.get_results()
    df["budget_int"] = df["num_simulations"].map(SBIBM_BUDGET_MAP)

    results: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for task in TASKS:
        results[task] = {}
        task_df = df[df["task"] == task]
        for budget_int in sorted(task_df["budget_int"].unique()):
            budget_df = task_df[task_df["budget_int"] == budget_int]
            results[task][int(budget_int)] = {}
            for algo in algorithms:
                algo_df = budget_df[budget_df["algorithm"] == algo]
                if algo_df.empty:
                    continue
                results[task][int(budget_int)][algo] = algo_df["C2ST"].values
    return results


def load_sbi_fresh_baselines(
    methods: list[str],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Load fresh sbi baseline results from outputs directory.

    Scans outputs/{task}/sbi_{method}_n{budget}/results/results.npz for each
    method and budget combination.

    Returns:
        {task_name: {budget: {"sbi_npe": array, "sbi_nle": array, ...}}}
    """
    results: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for task in TASKS:
        results[task] = {}
        for method in methods:
            for n in BUDGETS:
                npz_path = (
                    OUTPUTS_DIR
                    / task
                    / f"sbi_{method}_n{n}"
                    / "results"
                    / "results.npz"
                )
                if not npz_path.exists():
                    continue
                d = np.load(npz_path, allow_pickle=True)
                c2st_key = f"c2st_sbi_{method}"
                if c2st_key not in d:
                    continue
                if n not in results[task]:
                    results[task][n] = {}
                method_key = f"sbi_{method}"
                results[task][n][method_key] = d[c2st_key]
    return results


def merge_results(
    ours: dict[str, dict[int, dict]],
    baselines: dict[str, dict[int, dict]],
) -> dict[str, dict[int, dict]]:
    """Merge our results and SBIBM baselines into a single structure."""
    merged: dict[str, dict[int, dict]] = {}
    for task in TASKS:
        merged[task] = {}
        # Collect all budget points
        all_budgets = set()
        if task in ours:
            all_budgets.update(ours[task].keys())
        if task in baselines:
            all_budgets.update(baselines[task].keys())

        for n in sorted(all_budgets):
            merged[task][n] = {}
            if task in ours and n in ours[task]:
                merged[task][n].update(ours[task][n])
            if task in baselines and n in baselines[task]:
                merged[task][n].update(baselines[task][n])
    return merged


def main():
    parser = argparse.ArgumentParser(description="Generate budget sweep plots")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["NPE", "SNPE", "SNLE"],
        help="SBIBM algorithms to include (default: NPE SNPE SNLE)",
    )
    parser.add_argument(
        "--sbi-baselines",
        nargs="+",
        default=[],
        help="Fresh sbi methods to include (e.g., npe nle nre fmpe)",
    )
    args = parser.parse_args()

    print("Loading our results...")
    ours = load_our_results()
    tasks_with_data = [t for t in TASKS if ours.get(t)]
    for t in tasks_with_data:
        budgets = sorted(ours[t].keys())
        print(f"  {t}: budgets {budgets}")

    print(f"\nLoading SBIBM baselines: {args.baselines}")
    baselines = load_sbibm_baselines(args.baselines)

    sbi_fresh = {}
    if args.sbi_baselines:
        print(f"\nLoading fresh sbi baselines: {args.sbi_baselines}")
        sbi_fresh = load_sbi_fresh_baselines(args.sbi_baselines)
        for t in TASKS:
            if sbi_fresh.get(t):
                budgets = sorted(sbi_fresh[t].keys())
                methods_found = set()
                for b in budgets:
                    methods_found.update(sbi_fresh[t][b].keys())
                print(f"  {t}: {sorted(methods_found)} at budgets {budgets}")

    print("\nMerging results...")
    merged = merge_results(ours, baselines)
    # Merge fresh sbi results on top
    merged = merge_results(merged, sbi_fresh)

    # Define visual styles
    method_labels = {
        "tabpfn": "TabPFN (ours)",
        "raw": "Raw features",
    }
    method_styles = {
        "tabpfn": {"color": "#1f77b4", "linewidth": 2.5, "marker": "o", "markersize": 5},
        "raw": {"color": "#ff7f0e", "linewidth": 2.5, "marker": "s", "markersize": 5},
    }

    # SBIBM baseline styles — dashed lines, muted colors
    baseline_colors = {
        "NPE": "#2ca02c",
        "SNPE": "#d62728",
        "SNLE": "#9467bd",
        "NLE": "#8c564b",
        "NRE": "#e377c2",
        "SNRE": "#7f7f7f",
    }
    for algo in args.baselines:
        method_labels[algo] = f"{algo} (SBIBM)"
        method_styles[algo] = {
            "color": baseline_colors.get(algo, "gray"),
            "linewidth": 1.5,
            "linestyle": "--",
            "marker": "^",
            "markersize": 4,
            "alpha": 0.8,
        }

    # Fresh sbi baseline styles — solid lines, distinct colors
    sbi_fresh_colors = {
        "sbi_npe": "#17becf",
        "sbi_nle": "#bcbd22",
        "sbi_nre": "#e377c2",
        "sbi_fmpe": "#7f7f7f",
    }
    sbi_fresh_keys = []
    for m in args.sbi_baselines:
        key = f"sbi_{m}"
        sbi_fresh_keys.append(key)
        method_labels[key] = f"sbi {m.upper()} (fresh)"
        method_styles[key] = {
            "color": sbi_fresh_colors.get(key, "gray"),
            "linewidth": 2.0,
            "linestyle": "-",
            "marker": "D",
            "markersize": 4,
        }

    # Plot order: our methods first, then fresh sbi, then SBIBM precomputed
    methods = ["tabpfn", "raw"] + sbi_fresh_keys + args.baselines

    # Generate combined budget curves
    out_path = OUTPUTS_DIR / "budget_curves_with_baselines.png"
    print(f"\nGenerating {out_path}...")
    plot_budget_curves(
        merged,
        out_path,
        methods=methods,
        method_labels=method_labels,
        method_styles=method_styles,
    )

    # Also regenerate the standard plots (without baselines) for comparison
    out_path2 = OUTPUTS_DIR / "budget_curves.png"
    print(f"Regenerating {out_path2}...")
    plot_budget_curves(
        ours,
        out_path2,
        methods=["tabpfn", "raw"],
        method_labels={"tabpfn": "TabPFN (ours)", "raw": "Raw features"},
        method_styles={
            "tabpfn": {"color": "#1f77b4", "linewidth": 2.5, "marker": "o", "markersize": 5},
            "raw": {"color": "#ff7f0e", "linewidth": 2.5, "marker": "s", "markersize": 5},
        },
    )

    out_path3 = OUTPUTS_DIR / "budget_gap.png"
    print(f"Regenerating {out_path3}...")
    plot_budget_gap(ours, out_path3)

    print("\nDone!")


if __name__ == "__main__":
    main()
