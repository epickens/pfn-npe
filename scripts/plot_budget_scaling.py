"""C2ST vs simulation budget plotting utilities.

This script supports two workflows:
1) Legacy auto mode: infer tasks from disk and draw one multi-panel figure.
2) Paper variants: emit curated figures with stronger styling:
   - standard sbibm tasks (single row)
   - extended PFN-family stress tests
   - remaining extended tasks for appendix
   - standard + extended tasks (two rows)

Usage:
  uv run python scripts/plot_budget_scaling.py
  uv run python scripts/plot_budget_scaling.py --tasks slcp bernoulli_glm
  uv run python scripts/plot_budget_scaling.py --metric marginal
  uv run python scripts/plot_budget_scaling.py --min-budget-points 1
  uv run python scripts/plot_budget_scaling.py --min-methods 1
  uv run python scripts/plot_budget_scaling.py --make-paper-variants
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.plotting import plot_budget_curves  # noqa: E402

DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
DEFAULT_OUT = Path("pfn_testing/sbi/outputs/layer_ablation/figures/budget_scaling.png")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")
DEFAULT_BUDGET = 10000  # convention when filename has no `_n{N}` suffix
SBIBM_BUDGET_MAP = {"10³": 1000, "10⁴": 10000, "10⁵": 100000}

NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)(?:_(?P<rest>.+))?$")
N_SUFFIX_RE = re.compile(r"_n(\d+)$")

METHOD_LABELS = {
    "nsf": "PFN-NPE",
    "nsf_no_pca": "PFN-NPE (no PCA)",
    "npe_pfn": "NPE-PFN",
    "learned_summary_npe": "Learned-summary NPE",
    "NPE": "NPE (sbibm)",
    "NLE": "NLE (sbibm)",
    "SNPE": "SNPE (sbibm)",
    "SNLE": "SNLE (sbibm)",
    "fmpe": "FMPE",
    "copula_neural": "Cop-Neural",
    "copula": "Cop-Gauss",
    "residual_flow": "AR + flow",
    "pymc_hmc": "PyMC HMC",
}
METHOD_STYLES = {
    "nsf": {"color": "#0072B2", "marker": "o", "linewidth": 2.8, "markersize": 4.8},
    "nsf_no_pca": {
        "color": "#D55E00",
        "marker": "s",
        "linewidth": 2.2,
        "markersize": 4.4,
        "linestyle": ":",
    },
    "npe_pfn": {"color": "#222222", "marker": "D", "linewidth": 2.8, "markersize": 4.6},
    "learned_summary_npe": {
        "color": "#009E73",
        "marker": "^",
        "linewidth": 2.4,
        "markersize": 4.8,
        "linestyle": "-.",
    },
    "NPE": {
        "color": "#009E73",
        "marker": "^",
        "linewidth": 2.4,
        "markersize": 4.8,
        "linestyle": "--",
    },
    "NLE": {
        "color": "#E69F00",
        "marker": "v",
        "linewidth": 2.4,
        "markersize": 4.8,
        "linestyle": "--",
    },
    "SNPE": {
        "color": "#CC79A7",
        "marker": "P",
        "linewidth": 2.4,
        "markersize": 4.8,
        "linestyle": "--",
    },
    "SNLE": {
        "color": "#6A3D9A",
        "marker": "X",
        "linewidth": 2.4,
        "markersize": 4.8,
        "linestyle": "--",
    },
    "fmpe": {
        "color": "#CC79A7",
        "marker": "P",
        "linewidth": 2.4,
        "markersize": 4.8,
        "linestyle": "--",
    },
    "copula_neural": {"color": "#56B4E9", "marker": "v", "linewidth": 2.2, "markersize": 4.6},
    "copula": {
        "color": "#E69F00",
        "marker": "x",
        "linewidth": 2.0,
        "markersize": 4.6,
        "linestyle": ":",
    },
    "residual_flow": {"color": "#7F7F7F", "marker": "*", "linewidth": 2.0, "markersize": 5.2},
    "pymc_hmc": {"color": "#999999", "marker": "X", "linewidth": 2.0, "markersize": 4.8},
}

DEFAULT_METHODS = ["nsf", "learned_summary_npe", "our_ar", "npe_pfn", "copula_neural"]
PAPER_METHODS = ["nsf", "learned_summary_npe", "npe_pfn", "NPE", "NLE", "SNPE", "SNLE"]
METRIC_LABELS = {
    "joint": "Joint C2ST",
    "marginal": "Marginal C2ST",
    "rank": "Rank C2ST",
}

TASK_LABELS = {
    "two_moons": "Two moons",
    "gaussian_mixture": "Gauss. mix",
    "gaussian_linear": "Gauss. linear",
    "bernoulli_glm": "Bernoulli GLM",
    "slcp": "SLCP",
    "sir": "SIR",
    "lotka_volterra": "Lotka-Volterra",
    "ou": "OU",
    "two_moons_distractors": "Two moons (+ distr.)",
    "gaussian_mixture_distractors": "G. mix (+ distr.)",
    "bernoulli_glm_distractors": "Bern. GLM (+ distr.)",
    "sir_distractors": "SIR (+ distr.)",
    "ar1_ts_t50": "AR(1), T=50",
    "solar_dynamo": "Solar dynamo",
}
COMBINED_TASK_LABELS = {
    **TASK_LABELS,
    "two_moons_distractors": "Two moons\n(+ distr.)",
    "gaussian_mixture_distractors": "G. mix\n(+ distr.)",
    "bernoulli_glm_distractors": "Bern. GLM\n(+ distr.)",
    "sir_distractors": "SIR\n(+ distr.)",
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
DISTRACTOR_CORE_TASKS = [
    "two_moons_distractors",
    "gaussian_mixture_distractors",
    "bernoulli_glm_distractors",
    "sir_distractors",
]
HIGHDIM_TS_TASKS = [
    "ou",
    "ar1_ts_t50",
    "solar_dynamo",
]
STANDARD_PLUS_EXTENDED_TASKS = STANDARD_TASKS + DISTRACTOR_CORE_TASKS + HIGHDIM_TS_TASKS
EXTENDED_STRESS_TASKS = [
    "gaussian_mixture_distractors",
    "bernoulli_glm_distractors",
    "ou",
    "ar1_ts_t50",
]
APPENDIX_EXTENDED_TASKS = [
    "two_moons_distractors",
    "sir_distractors",
    "solar_dynamo",
]

PAPER_XTICKS = [100, 1000, 10000, 100000]
PAPER_XTICK_LABELS = [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"]


def parse_name(stem: str) -> tuple[str, int, str, int] | None:
    """Return (task, seed, method, n_train) from a c2st_decomp filename."""
    m = NAME_RE.match(stem)
    if not m:
        return None
    task, seed = m.group("task"), int(m.group("seed"))
    rest = m.group("rest") or ""
    n_match = N_SUFFIX_RE.search(rest)
    if n_match:
        n_train = int(n_match.group(1))
        method = rest[:n_match.start()]
    else:
        n_train = DEFAULT_BUDGET
        method = rest
    if not method:
        method = "nsf"
    return task, seed, method, n_train


def collect(
    metric: str,
    tasks: set[str] | None,
    methods: set[str] | None,
) -> dict[str, dict[int, dict[str, dict[int, float]]]]:
    """Read c2st_decomp npzs into nested dict keyed by task/budget/method/seed."""
    collected: dict[str, dict[int, dict[str, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for p in sorted(DECOMP_DIR.glob("*.npz")):
        parsed = parse_name(p.stem)
        if not parsed:
            continue
        task, seed, method, n_train = parsed
        if tasks is not None and task not in tasks:
            continue
        if methods is not None and method not in methods:
            continue
        try:
            d = np.load(p, allow_pickle=True)
            arr = d[metric]
        except (OSError, KeyError):
            continue
        collected[task][n_train][method][seed] = float(np.mean(arr))
    return collected


def build_budget_results(
    collected: dict[str, dict[int, dict[str, dict[int, float]]]],
    tasks_order: list[str] | None,
    methods: list[str],
    min_budget_points: int,
    min_methods: int,
    require_two_budgets: bool,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Convert collected dict into plot-ready arrays with filtering."""
    budget_results: dict[str, dict[int, dict[str, np.ndarray]]] = {}

    # Candidate tasks: explicit order if provided, else all available tasks.
    ordered_tasks = list(tasks_order) if tasks_order is not None else sorted(collected.keys())

    for task in ordered_tasks:
        if task not in collected:
            continue
        by_budget = collected[task]
        if require_two_budgets and len(by_budget) < 2:
            continue

        method_budget_counts = {
            method: sum(1 for methods_dict in by_budget.values() if method in methods_dict)
            for method in methods
        }

        filtered_by_budget = {
            n: {
                method: np.array(list(seed_means.values()), dtype=float)
                for method, seed_means in methods_dict.items()
                if method_budget_counts.get(method, 0) >= min_budget_points
            }
            for n, methods_dict in by_budget.items()
        }
        filtered_by_budget = {n: m for n, m in filtered_by_budget.items() if m}

        retained_methods = {
            method
            for methods_dict in filtered_by_budget.values()
            for method in methods_dict
        }
        if len(retained_methods) < min_methods:
            continue
        if require_two_budgets and len(filtered_by_budget) < 2:
            continue

        budget_results[task] = filtered_by_budget

    return budget_results


def merge_sbibm_baselines_into_collected(
    collected: dict[str, dict[int, dict[str, dict[int, float]]]],
    tasks: list[str],
    algorithms: list[str],
) -> None:
    """Add SBIBM-shipped baselines (NPE/NLE/etc.) to collected in-place."""
    try:
        import sbibm
    except Exception as exc:
        print(f"[warn] sbibm unavailable; skipping shipped baselines: {exc}")
        return

    try:
        df = sbibm.get_results()
    except Exception as exc:
        print(f"[warn] sbibm.get_results() failed; skipping shipped baselines: {exc}")
        return

    df = df[df["algorithm"].isin(algorithms)].copy()
    if df.empty:
        print(f"[warn] sbibm results have no rows for {algorithms}")
        return
    df["budget_int"] = df["num_simulations"].map(SBIBM_BUDGET_MAP)
    df = df[df["budget_int"].notna()]
    if df.empty:
        print("[warn] sbibm baseline rows lacked recognized budget keys")
        return

    for task in tasks:
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue
        for budget_int in sorted(task_df["budget_int"].astype(int).unique()):
            budget_df = task_df[task_df["budget_int"].astype(int) == int(budget_int)]
            for algo in algorithms:
                algo_df = budget_df[budget_df["algorithm"] == algo]
                if algo_df.empty:
                    continue
                # Use row index as pseudo-seed so downstream aggregation keeps
                # mean/std behavior consistent with local multi-seed curves.
                for row_idx, c2st in enumerate(algo_df["C2ST"].values):
                    collected[task][int(budget_int)][algo][row_idx] = float(c2st)


def write_curve_pair(
    budget_results: dict[str, dict[int, dict[str, np.ndarray]]],
    out_png: Path,
    methods: list[str],
    title: str | None,
    n_cols: int,
    legend_ncols: int,
    panel_size: tuple[float, float],
    box_aspect: float | None = 0.78,
    layout_rect: tuple[float, float, float, float] = (0, 0.18, 1, 0.92),
    subplot_wspace: float = 0.03,
    subplot_hspace: float = 0.12,
    x_label_bottom_row_only: bool = False,
    x_tick_bottom_row_only: bool = False,
    title_fontsize: float = 15,
    axis_labelsize: float = 11,
    tick_labelsize: float = 8,
    task_title_fontsize: float = 10,
    legend_fontsize: float = 10,
    bbox_inches: str | None = "tight",
    legend_bbox_y: float = -0.08,
    subplot_left: float | None = None,
    subplot_right: float | None = None,
    subplot_bottom: float | None = None,
    subplot_top: float | None = None,
    suptitle_y: float | None = None,
    task_labels: dict[str, str] | None = None,
) -> None:
    """Write matched PNG + PDF for a given budget-curve figure."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    common = dict(
        methods=methods,
        method_labels=METHOD_LABELS,
        method_styles=METHOD_STYLES,
        task_labels=task_labels or TASK_LABELS,
        title=title,
        xlabel="Budget",
        ylabel="C2ST",
        share_legend=True,
        legend_ncols=legend_ncols,
        n_cols=n_cols,
        panel_size=panel_size,
        box_aspect=box_aspect,
        sharey=True,
        xticks=PAPER_XTICKS,
        xtick_labels=PAPER_XTICK_LABELS,
        xlim=(80, 140000),
        y_label_first_col_only=True,
        x_label_first_col_only=True,
        x_label_bottom_row_only=x_label_bottom_row_only,
        x_tick_bottom_row_only=x_tick_bottom_row_only,
        title_fontsize=title_fontsize,
        axis_labelsize=axis_labelsize,
        tick_labelsize=tick_labelsize,
        task_title_fontsize=task_title_fontsize,
        title_fontweight="bold",
        axis_labelweight="bold",
        tick_labelweight="bold",
        task_title_fontweight="bold",
        legend_fontsize=legend_fontsize,
        line_alpha=0.78,
        grid_alpha=0.18,
        show_errorbars=False,
        legend_bbox_y=legend_bbox_y,
        layout_rect=layout_rect,
        subplot_wspace=subplot_wspace,
        subplot_hspace=subplot_hspace,
        subplot_left=subplot_left,
        subplot_right=subplot_right,
        subplot_bottom=subplot_bottom,
        subplot_top=subplot_top,
        suptitle_y=suptitle_y,
        bbox_inches=bbox_inches,
    )
    plot_budget_curves(budget_results, out_png, **common)
    out_pdf = out_png.with_suffix(".pdf")
    plot_budget_curves(budget_results, out_pdf, **common)
    print(f"Wrote {out_png}\nWrote {out_pdf}")


def print_summary(
    budget_results: dict[str, dict[int, dict[str, np.ndarray]]],
    methods: list[str],
    metric: str,
) -> None:
    """Console summary of retained tasks/budgets/method points."""
    print(f"Plotting {len(budget_results)} tasks ({metric} C2ST):")
    for task, by_budget in budget_results.items():
        budgets = sorted(by_budget.keys())
        method_counts = {
            m: sum(1 for b in budgets if m in by_budget[b])
            for m in methods
        }
        print(
            f"  {task:<32} budgets={budgets}  "
            f"per-method points: "
            + ", ".join(f"{m}={c}" for m, c in method_counts.items() if c)
        )


def make_paper_variants(metric: str) -> None:
    """Emit the curated main-text and appendix budget-scaling figures."""
    methods = PAPER_METHODS
    collected = collect(metric=metric, tasks=None, methods=set(methods))
    if metric == "joint":
        merge_sbibm_baselines_into_collected(
            collected=collected,
            tasks=STANDARD_PLUS_EXTENDED_TASKS,
            algorithms=["NPE", "NLE", "SNPE", "SNLE"],
        )
    else:
        print(f"[warn] sbibm shipped baselines are only joint C2ST; metric={metric} skips them")

    standard = build_budget_results(
        collected,
        tasks_order=STANDARD_TASKS,
        methods=methods,
        min_budget_points=1,
        min_methods=2,
        require_two_budgets=False,
    )
    if standard:
        print_summary(standard, methods, metric)
        write_curve_pair(
            standard,
            FIG_DIR / "budget_scaling_standard_row.png",
            methods,
            title=f"{METRIC_LABELS[metric]} vs Budget (standard sbibm tasks)",
            n_cols=len(STANDARD_TASKS),
            legend_ncols=3,
            panel_size=(1.24, 2.05),
            box_aspect=0.68,
            layout_rect=(0, 0.26, 1, 0.9),
            subplot_wspace=0.1,
        )

    standard_plus_extended = build_budget_results(
        collected,
        tasks_order=STANDARD_PLUS_EXTENDED_TASKS,
        methods=methods,
        min_budget_points=1,
        min_methods=2,
        require_two_budgets=False,
    )
    if standard_plus_extended:
        print_summary(standard_plus_extended, methods, metric)
        write_curve_pair(
            standard_plus_extended,
            FIG_DIR / "budget_scaling_standard_plus_extended.png",
            methods,
            title=None,
            n_cols=len(STANDARD_TASKS),
            legend_ncols=4,
            panel_size=(1.24, 1.82),
            box_aspect=0.68,
            layout_rect=(0, 0.14, 1, 0.99),
            subplot_wspace=0.14,
            subplot_hspace=0.26,
            x_label_bottom_row_only=True,
            x_tick_bottom_row_only=True,
            task_title_fontsize=8.5,
            task_labels=COMBINED_TASK_LABELS,
            legend_bbox_y=-0.02,
        )

    extended = build_budget_results(
        collected,
        tasks_order=EXTENDED_STRESS_TASKS,
        methods=methods,
        min_budget_points=1,
        min_methods=2,
        require_two_budgets=False,
    )
    if extended:
        print_summary(extended, methods, metric)
        write_curve_pair(
            extended,
            FIG_DIR / "budget_scaling_extended_stress.png",
            methods,
            title=f"{METRIC_LABELS[metric]} vs Budget (extended PFN stress tests)",
            n_cols=len(EXTENDED_STRESS_TASKS),
            legend_ncols=2,
            panel_size=(1.18, 2.05),
            box_aspect=0.72,
            layout_rect=(0, 0.27, 1, 0.96),
            subplot_wspace=0.26,
            title_fontsize=9,
            axis_labelsize=7,
            tick_labelsize=6,
            task_title_fontsize=7,
            legend_fontsize=7,
            bbox_inches=None,
            legend_bbox_y=0.16,
            subplot_left=0.12,
            subplot_right=0.98,
            subplot_bottom=0.35,
            subplot_top=0.73,
            suptitle_y=0.9,
        )

    appendix = build_budget_results(
        collected,
        tasks_order=APPENDIX_EXTENDED_TASKS,
        methods=methods,
        min_budget_points=1,
        min_methods=2,
        require_two_budgets=False,
    )
    if appendix:
        print_summary(appendix, methods, metric)
        write_curve_pair(
            appendix,
            FIG_DIR / "budget_scaling_appendix_extended.png",
            methods,
            title=f"{METRIC_LABELS[metric]} vs Budget (additional extended tasks)",
            n_cols=len(APPENDIX_EXTENDED_TASKS),
            legend_ncols=2,
            # Match the extended-stress figure's 4.72 x 2.05 in canvas.
            panel_size=(4.72 / len(APPENDIX_EXTENDED_TASKS), 2.05),
            box_aspect=0.72,
            layout_rect=(0, 0.27, 1, 0.96),
            subplot_wspace=0.26,
            title_fontsize=9,
            axis_labelsize=7,
            tick_labelsize=6,
            task_title_fontsize=7,
            legend_fontsize=7,
            bbox_inches=None,
            legend_bbox_y=0.16,
            subplot_left=0.12,
            subplot_right=0.98,
            subplot_bottom=0.35,
            subplot_top=0.73,
            suptitle_y=0.9,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Restrict tasks. Default: all with ≥2 budgets on disk.")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--metric", default="joint",
                    choices=["joint", "marginal", "rank"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-budget-points", type=int, default=2,
                    help="Drop method/task series with fewer budget points. "
                         "Default 2 keeps the figure to actual curves; use 1 "
                         "to include one-off comparison points.")
    ap.add_argument("--min-methods", type=int, default=2,
                    help="Drop tasks with fewer retained method curves. "
                         "Default 2 keeps the main figure comparative; use 1 "
                         "for exploratory single-method budget panels.")
    ap.add_argument(
        "--make-paper-variants",
        action="store_true",
        help="Also emit curated standard, extended-stress, and appendix figures.",
    )
    args = ap.parse_args()

    if not DECOMP_DIR.exists():
        sys.exit(f"Missing decomp dir: {DECOMP_DIR}")

    if args.make_paper_variants:
        make_paper_variants(metric=args.metric)
        return

    collected = collect(
        metric=args.metric,
        tasks=set(args.tasks) if args.tasks is not None else None,
        methods=set(args.methods),
    )

    budget_results = build_budget_results(
        collected,
        tasks_order=args.tasks,
        methods=args.methods,
        min_budget_points=args.min_budget_points,
        min_methods=args.min_methods,
        require_two_budgets=True,
    )

    if not budget_results:
        sys.exit("No tasks have ≥2 budget points. Need budget-suffixed cells.")

    print_summary(budget_results, args.methods, args.metric)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_budget_curves(
        budget_results,
        out,
        methods=args.methods,
        method_labels=METHOD_LABELS,
        method_styles=METHOD_STYLES,
        task_labels=TASK_LABELS,
        title=f"{METRIC_LABELS[args.metric]} vs Simulation Budget",
        xlabel="Training simulations",
        ylabel="C2ST (lower is better)",
        share_legend=True,
        legend_ncols=min(len(args.methods), 4),
    )

    pdf_path = out.with_suffix(".pdf")
    plot_budget_curves(
        budget_results,
        pdf_path,
        methods=args.methods,
        method_labels=METHOD_LABELS,
        method_styles=METHOD_STYLES,
        task_labels=TASK_LABELS,
        title=f"{METRIC_LABELS[args.metric]} vs Simulation Budget",
        xlabel="Training simulations",
        ylabel="C2ST (lower is better)",
        share_legend=True,
        legend_ncols=min(len(args.methods), 4),
    )
    print(f"Wrote {out}\nWrote {pdf_path}")


if __name__ == "__main__":
    main()
