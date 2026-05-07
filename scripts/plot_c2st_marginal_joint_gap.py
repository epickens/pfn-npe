"""All-task marginal-vs-joint C2ST gap summary for PFN-NPE.

Reads decomposition outputs from `c2st_decomp/` at the 10k budget for the
vanilla PFN-NPE NSF head, then writes:
  - an all-task marginal-vs-joint gap figure
  - a companion LaTeX table with joint, marginal, rank, and joint-marginal gap
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tex_table import write_tex_tabular  # noqa: E402

DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")
TABLE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/tables")
CSV_OUT = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_marginal_joint_gap.csv")

DEFAULT_BUDGET = 10000
METHOD = "nsf"
NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)(?:_(?P<rest>.+))?$")
N_SUFFIX_RE = re.compile(r"_n(\d+)$")

TASK_LABELS = {
    "two_moons": "Two moons",
    "gaussian_mixture": "Gaussian mixture",
    "gaussian_linear": "Gaussian linear",
    "gaussian_linear_uniform": "Gaussian linear uniform",
    "bernoulli_glm": "Bernoulli GLM",
    "sir": "SIR",
    "lotka_volterra": "Lotka-Volterra",
    "slcp": "SLCP",
    "slcp_distractors": "SLCP + distractors",
    "two_moons_distractors": "Two moons + distractors",
    "gaussian_mixture_distractors": "Gaussian mixture + distractors",
    "bernoulli_glm_distractors": "Bernoulli GLM + distractors",
    "sir_distractors": "SIR + distractors",
    "ar1_ts_t50": "AR(1) time series",
    "ou": "Ornstein-Uhlenbeck",
    "solar_dynamo": "Solar dynamo",
}
TASK_ORDER = list(TASK_LABELS)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_name(stem: str) -> tuple[str, int, str, int] | None:
    match = NAME_RE.match(stem)
    if not match:
        return None
    rest = match.group("rest") or ""
    budget_match = N_SUFFIX_RE.search(rest)
    if budget_match:
        budget = int(budget_match.group(1))
        method = rest[: budget_match.start()]
    else:
        budget = DEFAULT_BUDGET
        method = rest or METHOD
    return match.group("task"), int(match.group("seed")), method, budget


def load_seed_rows() -> dict[str, dict[int, dict[str, float]]]:
    """Return data[task][seed] = metric means over reference observations."""
    data: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for path in sorted(DECOMP_DIR.glob("*.npz")):
        parsed = parse_name(path.stem)
        if parsed is None:
            continue
        task, seed, method, budget = parsed
        if budget != DEFAULT_BUDGET or method != METHOD:
            continue
        if task not in TASK_LABELS:
            continue
        loaded = np.load(path, allow_pickle=True)
        data[task][seed] = {
            "joint": float(np.mean(loaded["joint"])),
            "marginal": float(np.mean(loaded["marginal"])),
            "rank": float(np.mean(loaded["rank"])),
        }
    return data


def aggregate(data: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in TASK_ORDER:
        seed_rows = data.get(task, {})
        if not seed_rows:
            continue
        joint = np.asarray([row["joint"] for row in seed_rows.values()], dtype=float)
        marginal = np.asarray([row["marginal"] for row in seed_rows.values()], dtype=float)
        rank = np.asarray([row["rank"] for row in seed_rows.values()], dtype=float)
        gap = joint - marginal
        rows.append(
            {
                "task": task,
                "label": TASK_LABELS[task],
                "n_seeds": len(seed_rows),
                "joint_mean": float(joint.mean()),
                "joint_sd": float(joint.std()),
                "marginal_mean": float(marginal.mean()),
                "marginal_sd": float(marginal.std()),
                "rank_mean": float(rank.mean()),
                "rank_sd": float(rank.std()),
                "gap_mean": float(gap.mean()),
                "gap_sd": float(gap.std()),
            }
        )
    return sorted(rows, key=lambda row: float(row["gap_mean"]))


def fmt_pm(mean: float, sd: float) -> str:
    return f"{mean:.3f} $\\pm$ {sd:.2f}"


def write_outputs(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

    csv_lines = [
        "task,n_seeds,joint_mean,joint_sd,marginal_mean,marginal_sd,"
        "rank_mean,rank_sd,gap_mean,gap_sd"
    ]
    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    str(row["task"]),
                    str(row["n_seeds"]),
                    f"{row['joint_mean']:.6f}",
                    f"{row['joint_sd']:.6f}",
                    f"{row['marginal_mean']:.6f}",
                    f"{row['marginal_sd']:.6f}",
                    f"{row['rank_mean']:.6f}",
                    f"{row['rank_sd']:.6f}",
                    f"{row['gap_mean']:.6f}",
                    f"{row['gap_sd']:.6f}",
                ]
            )
        )
    CSV_OUT.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Wrote {CSV_OUT}")

    tex_rows = []
    for row in rows:
        tex_rows.append(
            [
                str(row["label"]),
                str(row["n_seeds"]),
                fmt_pm(float(row["joint_mean"]), float(row["joint_sd"])),
                fmt_pm(float(row["marginal_mean"]), float(row["marginal_sd"])),
                fmt_pm(float(row["rank_mean"]), float(row["rank_sd"])),
                fmt_pm(float(row["gap_mean"]), float(row["gap_sd"])),
            ]
        )
    write_tex_tabular(
        out_path=TABLE_DIR / "c2st_marginal_joint_gap.tex",
        columns=[
            "Task",
            "$n$",
            "Joint",
            "Marginal",
            "Rank",
            "Joint - marg.",
        ],
        rows=tex_rows,
        column_align="lrrrrr",
        source_script="scripts/plot_c2st_marginal_joint_gap.py",
    )


def plot_gap(rows: list[dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in rows]
    joint = np.asarray([row["joint_mean"] for row in rows], dtype=float)
    marginal = np.asarray([row["marginal_mean"] for row in rows], dtype=float)
    gap = joint - marginal
    y = np.arange(len(rows))
    n_runs = sum(int(row["n_seeds"]) for row in rows)

    fig_height = max(6.2, 0.36 * len(rows) + 1.7)
    fig, ax = plt.subplots(figsize=(8.4, fig_height))
    ax.axvline(0.5, color="#666666", linestyle=":", linewidth=1.2, label="ideal C2ST")
    for yi, marg, jnt, gp in zip(y, marginal, joint, gap, strict=True):
        color = "#D55E00" if gp > 0 else "#747474"
        ax.hlines(yi, marg, jnt, color=color, linewidth=2.2, alpha=0.8, zorder=1)
    ax.scatter(marginal, y, color="#009E73", s=55, edgecolor="white", linewidth=0.8,
               label="marginal", zorder=3)
    ax.scatter(joint, y, color="#0072B2", s=55, edgecolor="white", linewidth=0.8,
               label="joint", zorder=3)

    for yi, jnt, gp in zip(y, joint, gap, strict=True):
        if abs(gp) < 0.02:
            continue
        ax.text(
            min(jnt + 0.012, 0.985),
            yi,
            f"+{gp:.2f}",
            va="center",
            ha="left" if jnt < 0.94 else "right",
            fontsize=8.2,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.45, 1.01)
    ax.set_xlabel("C2ST against reference posterior samples")
    ax.set_title("PFN-NPE joint errors often exceed marginal errors")
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, ncol=3)
    fig.text(
        0.5,
        0.945,
        f"{len(rows)} tasks, {n_runs} task-seed runs at 10k simulations; "
        "segments show joint - marginal C2ST.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.08, top=0.89)

    out_png = FIG_DIR / "c2st_decomposition_summary.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main() -> None:
    rows = aggregate(load_seed_rows())
    if not rows:
        raise SystemExit("No matching PFN-NPE decomposition rows found.")
    write_outputs(rows)
    plot_gap(rows)

    print("\n=== PFN-NPE marginal-vs-joint C2ST gap ===")
    print(f"{'task':<34} {'n':>2} {'joint':>7} {'marg':>7} {'rank':>7} {'gap':>7}")
    for row in reversed(rows):
        print(
            f"{row['task']:<34} {row['n_seeds']:>2} "
            f"{row['joint_mean']:>7.3f} {row['marginal_mean']:>7.3f} "
            f"{row['rank_mean']:>7.3f} {row['gap_mean']:>+7.3f}"
        )


if __name__ == "__main__":
    main()
