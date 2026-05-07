"""Cross-theta probe case-study figures.

This script uses the saved cross-theta probe tensors to visualize two related
questions without collapsing them into a heatmap:

1. Which parameters become linearly decodable across encoder layers?
2. Does prompting TabPFN with a parameter as the target make that same
   parameter more accessible than it is from nonmatching target chunks?

Reads:
  - pfn_testing/sbi/outputs/layer_ablation/cross_theta/slcp_s*.npz
  - pfn_testing/sbi/outputs/layer_ablation/cross_theta/slcp_distractors_s*.npz
  - pfn_testing/sbi/outputs/layer_ablation/cross_theta/ar1_ts_t50_s*.npz

Writes:
  - pfn_testing/sbi/outputs/layer_ablation/figures/slcp_probe_target_accessibility.{png,pdf}
  - pfn_testing/sbi/outputs/layer_ablation/figures/ar1_probe_target_accessibility.{png,pdf}
  - pfn_testing/sbi/outputs/layer_ablation/tables/slcp_probe_target_accessibility.tex
  - pfn_testing/sbi/outputs/layer_ablation/tables/ar1_probe_target_accessibility.tex
  - pfn_testing/sbi/outputs/layer_ablation/*_probe_target_accessibility.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tex_table import write_tex_tabular  # noqa: E402

CROSS_THETA_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/cross_theta")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")
TABLE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/tables")

CASE_STUDIES = {
    "slcp": {
        "tasks": {
            "slcp": "SLCP",
            "slcp_distractors": "SLCP + distractors",
        },
        "param_labels": [
            r"$\theta_0$: mean 1",
            r"$\theta_1$: mean 2",
            r"$\theta_2$: scale 1",
            r"$\theta_3$: scale 2",
            r"$\theta_4$: corr.",
        ],
        "param_labels_table": [
            r"$\theta_0$ mean 1",
            r"$\theta_1$ mean 2",
            r"$\theta_2$ scale 1",
            r"$\theta_3$ scale 2",
            r"$\theta_4$ corr.",
        ],
        "param_colors": ["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73"],
        "title": "SLCP parameter-specific probe",
        "out_stem": "slcp_probe_target_accessibility",
        "figsize": (7.2, 10.6),
    },
    "ar1": {
        "tasks": {
            "ar1_ts_t50": "AR(1)",
        },
        "param_labels": [
            r"$\rho$: autocorr.",
            r"$\log\sigma$: innovation scale",
        ],
        "param_labels_table": [
            r"$\rho$ autocorr.",
            r"$\log\sigma$ innovation scale",
        ],
        "param_colors": ["#0072B2", "#D55E00"],
        "title": "AR(1) parameter-specific probe",
        "out_stem": "ar1_probe_target_accessibility",
        "figsize": (4.6, 4.8),
    },
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_task(task: str, dim_theta: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return layers, R2 array with shape (seed, layer, source, decoded), seeds."""
    runs: list[np.ndarray] = []
    seeds: list[int] = []
    layers_ref: np.ndarray | None = None
    name_re = re.compile(rf"{re.escape(task)}_s(?P<seed>\d+)\.npz$")
    for path in sorted(CROSS_THETA_DIR.glob(f"{task}_s*.npz")):
        match = name_re.match(path.name)
        if match is None:
            continue
        loaded = np.load(path, allow_pickle=True)
        layers = np.asarray(loaded["layers"], dtype=int)
        r2 = np.asarray(loaded["R2"], dtype=float)
        if r2.shape[1:] != (dim_theta, dim_theta):
            raise ValueError(
                f"Expected {dim_theta}x{dim_theta} R2 matrices in {path}, got {r2.shape}"
            )
        if layers_ref is None:
            layers_ref = layers
        elif not np.array_equal(layers_ref, layers):
            raise ValueError(f"Layer mismatch in {path}")
        runs.append(r2)
        seeds.append(int(loaded["seed"]))
    if layers_ref is None or not runs:
        raise FileNotFoundError(f"No cross-theta runs for {task} in {CROSS_THETA_DIR}")
    return layers_ref, np.stack(runs), seeds


def off_target_values(r2: np.ndarray, decoded_dim: int) -> np.ndarray:
    """Return R2 values for all nonmatching source chunks for one decoded dim."""
    sources = [i for i in range(r2.shape[-2]) if i != decoded_dim]
    return r2[..., sources, decoded_dim]


def plot_case_study(
    case: dict[str, object],
    data: dict[str, tuple[np.ndarray, np.ndarray, list[int]]],
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = case["tasks"]
    param_labels = case["param_labels"]
    param_colors = case["param_colors"]
    fig, axes = plt.subplots(
        len(param_labels),
        len(tasks),
        figsize=case["figsize"],
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for col_idx, (task, task_label) in enumerate(tasks.items()):
        layers, r2_runs, seeds = data[task]
        r2_mean = r2_runs.mean(axis=0)
        r2_sd = r2_runs.std(axis=0)
        for decoded_dim, param_label in enumerate(param_labels):
            ax = axes[decoded_dim, col_idx]
            color = param_colors[decoded_dim]
            matched_mean = r2_mean[:, decoded_dim, decoded_dim]
            matched_sd = r2_sd[:, decoded_dim, decoded_dim]

            off_mean_by_seed = off_target_values(r2_runs, decoded_dim).mean(axis=-1)
            off_mean = off_mean_by_seed.mean(axis=0)
            off_sd = off_mean_by_seed.std(axis=0)
            off_individual = off_target_values(r2_mean, decoded_dim)

            for source_idx in range(off_individual.shape[1]):
                ax.plot(
                    layers,
                    off_individual[:, source_idx],
                    color="0.72",
                    lw=0.7,
                    alpha=0.65,
                    zorder=1,
                )
            ax.plot(
                layers,
                off_mean,
                color="0.20",
                lw=1.4,
                ls="--",
                zorder=2,
            )
            ax.fill_between(
                layers,
                off_mean - off_sd,
                off_mean + off_sd,
                color="0.50",
                alpha=0.12,
                linewidth=0,
                zorder=0,
            )
            ax.plot(
                layers,
                matched_mean,
                color=color,
                lw=2.2,
                marker="o",
                ms=3,
                zorder=3,
            )
            ax.fill_between(
                layers,
                matched_mean - matched_sd,
                matched_mean + matched_sd,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            ax.axhline(0.0, color="0.82", lw=0.7, ls=":")
            ax.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.35, zorder=-1)
            ax.set_xlim(float(layers.min()) - 0.3, float(layers.max()) + 0.3)
            ax.set_ylim(-0.08, 1.02)
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            if decoded_dim == 0:
                ax.set_title(task_label)
            if col_idx == 0:
                ax.set_ylabel(f"{param_label}\nRidge val $R^2$")
            if decoded_dim == len(param_labels) - 1:
                ax.set_xlabel("Encoder layer")
            if decoded_dim == 0 and col_idx == 0:
                ax.text(
                    0.03,
                    0.95,
                    f"n={len(seeds)} seeds",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="0.25",
                )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            lw=2.2,
            marker="o",
            ms=3,
            label=r"matching target chunk ($e_j \to \theta_j$)",
        ),
        mlines.Line2D(
            [],
            [],
            color="0.20",
            lw=1.4,
            ls="--",
            label=r"mean nonmatching chunks ($e_i \to \theta_j$, $i \neq j$)",
        ),
        mlines.Line2D(
            [],
            [],
            color="0.72",
            lw=0.9,
            label="individual nonmatching chunks",
        ),
        mlines.Line2D(
            [],
            [],
            color="#FFF3CD",
            lw=8,
            alpha=0.7,
            label="layer 4 transition region",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        str(case["title"]),
        y=0.995,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.985))
    for suffix in ("png", "pdf"):
        out = FIG_DIR / f"{case['out_stem']}.{suffix}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def fmt_pm(mean: float, sd: float) -> str:
    return f"{mean:.2f} $\\pm$ {sd:.2f}"


def write_summary_table(
    case: dict[str, object],
    data: dict[str, tuple[np.ndarray, np.ndarray, list[int]]],
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = case["tasks"]
    param_labels_table = case["param_labels_table"]
    csv_out = Path("pfn_testing/sbi/outputs/layer_ablation") / f"{case['out_stem']}.csv"
    rows: list[list[str]] = []
    csv_lines = [
        "task,decoded_dim,parameter,matched_l11_mean,matched_l11_sd,"
        "off_mean_l11_mean,off_mean_l11_sd,off_max_l11_mean,off_max_l11_sd,"
        "matched_minus_off_mean,matched_minus_off_sd"
    ]
    midrules_after: list[int] = []
    for task_idx, (task, task_label) in enumerate(tasks.items()):
        _, r2_runs, _ = data[task]
        final = r2_runs[:, -1]
        for decoded_dim, param_label in enumerate(param_labels_table):
            matched = final[:, decoded_dim, decoded_dim]
            off = off_target_values(final, decoded_dim)
            off_mean = off.mean(axis=1)
            off_max = off.max(axis=1)
            delta = matched - off_mean
            rows.append(
                [
                    task_label,
                    param_label,
                    fmt_pm(float(matched.mean()), float(matched.std())),
                    fmt_pm(float(off_mean.mean()), float(off_mean.std())),
                    fmt_pm(float(off_max.mean()), float(off_max.std())),
                    fmt_pm(float(delta.mean()), float(delta.std())),
                ]
            )
            csv_lines.append(
                ",".join(
                    [
                        task,
                        str(decoded_dim),
                        param_label.replace(",", ""),
                        f"{matched.mean():.6f}",
                        f"{matched.std():.6f}",
                        f"{off_mean.mean():.6f}",
                        f"{off_mean.std():.6f}",
                        f"{off_max.mean():.6f}",
                        f"{off_max.std():.6f}",
                        f"{delta.mean():.6f}",
                        f"{delta.std():.6f}",
                    ]
                )
            )
        if task_idx == 0:
            midrules_after.append(len(rows) - 1)

    csv_out.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_out}")
    write_tex_tabular(
        out_path=TABLE_DIR / f"{case['out_stem']}.tex",
        columns=[
            "Task",
            "Decoded parameter",
            "Matched",
            "Off mean",
            "Off max",
            "$\\Delta$",
        ],
        rows=rows,
        column_align="llrrrr",
        midrules_after=midrules_after,
        source_script="scripts/plot_slcp_probe_case_study.py",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=[*CASE_STUDIES, "all"], default="slcp")
    args = ap.parse_args()

    cases = CASE_STUDIES if args.case == "all" else {args.case: CASE_STUDIES[args.case]}
    for case_name, case in cases.items():
        dim_theta = len(case["param_labels"])
        data = {task: load_task(task, dim_theta) for task in case["tasks"]}
        plot_case_study(case, data)
        write_summary_table(case, data)

        print(f"\nFinal-layer matched vs nonmatching target chunks: {case_name}")
        for task, (_, r2_runs, seeds) in data.items():
            print(f"  {case['tasks'][task]} ({len(seeds)} seeds)")
            final = r2_runs[:, -1]
            for decoded_dim, param_label in enumerate(case["param_labels_table"]):
                matched = final[:, decoded_dim, decoded_dim]
                off_mean = off_target_values(final, decoded_dim).mean(axis=1)
                print(
                    f"    {param_label:<30} matched={matched.mean():+.3f} "
                    f"off_mean={off_mean.mean():+.3f} "
                    f"delta={np.mean(matched - off_mean):+.3f}"
                )


if __name__ == "__main__":
    main()
