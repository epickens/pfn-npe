"""Main probe summary figure and parameter-role appendix table.

The main figure replaces the separate mean/variance probe plots with a compact
three-panel diagnostic:
  A. all-task mean-theta linear probe summary,
  B. SLCP matched-target decoding by semantic parameter role,
  C. SLCP+distractors matched-target decoding by semantic parameter role.

The appendix table summarizes final-layer matched and off-target decoding for
interpretable task parameters.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tex_table import write_tex_tabular  # noqa: E402

ABL_DIR = Path("pfn_testing/sbi/outputs/layer_ablation")
PROBE_DIR = ABL_DIR / "probe"
CROSS_THETA_DIR = ABL_DIR / "cross_theta"
FIG_DIR = ABL_DIR / "figures"
TABLE_DIR = ABL_DIR / "tables"
ROLE_CSV_OUT = ABL_DIR / "probe_parameter_role_summary.csv"

NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)\.npz$")
N_LAYERS = 12

TASK_LABELS = {
    "ar1_ts_t50": "AR(1)",
    "bernoulli_glm": "Bernoulli GLM",
    "bernoulli_glm_distractors": "Bernoulli GLM + distr.",
    "gaussian_linear_uniform": "Gaussian linear uniform",
    "gaussian_mixture": "Gaussian mixture",
    "gaussian_mixture_distractors": "Gaussian mixture + distr.",
    "lotka_volterra": "Lotka-Volterra",
    "ou": "OU",
    "sir": "SIR",
    "sir_distractors": "SIR + distr.",
    "slcp": "SLCP",
    "slcp_distractors": "SLCP + distr.",
    "solar_dynamo": "Solar dynamo",
    "two_moons": "Two moons",
    "two_moons_distractors": "Two moons + distr.",
}
DISTRACTOR_TASKS = {
    "two_moons_distractors",
    "gaussian_mixture_distractors",
    "sir_distractors",
    "slcp_distractors",
    "bernoulli_glm_distractors",
}
HIGHDIM_TASKS = {"ou", "solar_dynamo", "ar1_ts_t50", "lotka_volterra"}
GROUP_COLORS = {"distr": "#0072B2", "standard": "#009E73", "highdim": "#D55E00"}
GROUP_LABELS = {
    "distr": "distractor variants",
    "standard": "standard SBIBM",
    "highdim": "high-dim/time-series",
}

SLCP_PARAM_LABELS = [
    r"$\theta_0$ mean 1",
    r"$\theta_1$ mean 2",
    r"$\theta_2$ scale 1",
    r"$\theta_3$ scale 2",
    r"$\theta_4$ corr.",
]
SLCP_COLORS = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73"]

ROLE_TASKS: dict[str, list[tuple[str, str, str]]] = {
    "slcp": [
        (r"$\theta_0$", "mean 1", "location"),
        (r"$\theta_1$", "mean 2", "location"),
        (r"$\theta_2$", "scale 1", "scale"),
        (r"$\theta_3$", "scale 2", "scale"),
        (r"$\theta_4$", "corr.", "correlation"),
    ],
    "slcp_distractors": [
        (r"$\theta_0$", "mean 1", "location"),
        (r"$\theta_1$", "mean 2", "location"),
        (r"$\theta_2$", "scale 1", "scale"),
        (r"$\theta_3$", "scale 2", "scale"),
        (r"$\theta_4$", "corr.", "correlation"),
    ],
    "ou": [
        (r"$\alpha$", "long-run mean", "location"),
        (r"$\beta$", "reversion rate", "rate"),
        (r"$\sigma$", "diffusion scale", "scale"),
    ],
    "solar_dynamo": [
        (r"$\alpha_{\min}$", "growth floor", "location/rate"),
        (r"$\alpha_{\mathrm{range}}$", "growth range", "scale/range"),
        (r"$\epsilon_{\max}$", "noise bound", "scale"),
    ],
    "ar1_ts_t50": [
        (r"$\rho$", "autocorrelation", "correlation/rate"),
        (r"$\log\sigma$", "innovation scale", "scale"),
    ],
    "sir": [
        (r"$\beta$", "infection rate", "rate"),
        (r"$\gamma$", "recovery rate", "rate"),
    ],
    "lotka_volterra": [
        (r"$\alpha$", "prey growth", "rate"),
        (r"$\beta$", "predation", "interaction"),
        (r"$\gamma$", "predator death", "rate"),
        (r"$\delta$", "predator growth", "interaction"),
    ],
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
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


def task_group(task: str) -> str:
    if task in DISTRACTOR_TASKS:
        return "distr"
    if task in HIGHDIM_TASKS:
        return "highdim"
    return "standard"


def fmt_pm(mean: float, sd: float) -> str:
    return f"{mean:.2f} $\\pm$ {sd:.2f}"


def load_mean_probe_data() -> dict[str, dict[str, object]]:
    by_task: dict[str, list[dict[str, np.ndarray | int]]] = {}
    for path in sorted(PROBE_DIR.glob("*.npz")):
        match = NAME_RE.match(path.name)
        if match is None:
            continue
        task = match.group("task")
        if task not in TASK_LABELS:
            continue
        loaded = np.load(path, allow_pickle=True)
        by_task.setdefault(task, []).append(
            {
                "r2": np.asarray(loaded["r2"], dtype=float),
                "seed": int(loaded["seed"]),
            }
        )

    out: dict[str, dict[str, object]] = {}
    for task, runs in by_task.items():
        r2_arr = np.stack([np.asarray(run["r2"], dtype=float) for run in runs])
        out[task] = {
            "r2_mean": r2_arr.mean(axis=0),
            "r2_sd": r2_arr.std(axis=0),
            "n_seeds": len(runs),
        }
    return out


def load_cross_theta(task: str) -> tuple[np.ndarray, np.ndarray]:
    runs: list[np.ndarray] = []
    layers_ref: np.ndarray | None = None
    for path in sorted(CROSS_THETA_DIR.glob(f"{task}_s*.npz")):
        if "_mv25" in path.name or "_ls" in path.name:
            continue
        match = NAME_RE.match(path.name)
        if match is None or match.group("task") != task:
            continue
        loaded = np.load(path, allow_pickle=True)
        layers = np.asarray(loaded["layers"], dtype=int)
        if layers_ref is None:
            layers_ref = layers
        elif not np.array_equal(layers_ref, layers):
            raise ValueError(f"Layer mismatch in {path}")
        runs.append(np.asarray(loaded["R2"], dtype=float))
    if layers_ref is None or not runs:
        raise FileNotFoundError(f"No cross-theta runs for {task}")
    return layers_ref, np.stack(runs)


def plot_overall_probe(ax: plt.Axes, data: dict[str, dict[str, object]]) -> None:
    layers = np.arange(N_LAYERS)
    by_group: dict[str, list[np.ndarray]] = {"distr": [], "standard": [], "highdim": []}
    for task, task_data in data.items():
        curve = np.asarray(task_data["r2_mean"], dtype=float)
        group = task_group(task)
        by_group[group].append(curve)
        ax.plot(layers, curve, color=GROUP_COLORS[group], lw=0.7, alpha=0.22)

    for group in ("standard", "distr", "highdim"):
        curves = by_group[group]
        if not curves:
            continue
        arr = np.stack(curves)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0)
        ax.plot(
            layers,
            mean,
            color=GROUP_COLORS[group],
            lw=2.1,
            label=f"{GROUP_LABELS[group]} (n={len(curves)})",
        )
        ax.fill_between(layers, mean - sd, mean + sd, color=GROUP_COLORS[group], alpha=0.13)

    ax.axhline(0, color="0.75", lw=0.7, ls=":")
    ax.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.35, zorder=-1)
    ax.set_title("A. Mean-parameter decoding across tasks")
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel(r"Ridge val $R^2$")
    ax.set_xlim(-0.3, 11.3)
    ax.set_ylim(-0.08, 1.02)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=7)


def plot_slcp_matched(ax: plt.Axes, task: str, title: str) -> None:
    layers, r2_runs = load_cross_theta(task)
    diag = np.stack([np.diagonal(run, axis1=1, axis2=2) for run in r2_runs])
    diag_mean = diag.mean(axis=0)
    diag_sd = diag.std(axis=0)
    for dim, label in enumerate(SLCP_PARAM_LABELS):
        ax.plot(layers, diag_mean[:, dim], lw=2.0, marker="o", ms=3, color=SLCP_COLORS[dim], label=label)
        ax.fill_between(
            layers,
            diag_mean[:, dim] - diag_sd[:, dim],
            diag_mean[:, dim] + diag_sd[:, dim],
            color=SLCP_COLORS[dim],
            alpha=0.14,
            linewidth=0,
        )
    ax.axhline(0, color="0.75", lw=0.7, ls=":")
    ax.axvspan(3.5, 4.5, color="#FFF3CD", alpha=0.35, zorder=-1)
    ax.set_title(title)
    ax.set_xlabel("Encoder layer")
    ax.set_xlim(-0.3, 11.3)
    ax.set_ylim(-0.08, 1.02)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))


def plot_main_figure() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    probe_data = load_mean_probe_data()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharey=True)
    plot_overall_probe(axes[0], probe_data)
    plot_slcp_matched(axes[1], "slcp", "B. SLCP matched-target decoding")
    plot_slcp_matched(axes[2], "slcp_distractors", "C. SLCP + distractors")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    handles = [
        mlines.Line2D([], [], color=SLCP_COLORS[i], lw=2.0, marker="o", ms=3, label=label)
        for i, label in enumerate(SLCP_PARAM_LABELS)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.62, -0.035),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for suffix in ("png", "pdf"):
        out = FIG_DIR / f"probe_main_summary.{suffix}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def off_target_values(r2: np.ndarray, decoded_dim: int) -> np.ndarray:
    sources = [idx for idx in range(r2.shape[-2]) if idx != decoded_dim]
    return r2[..., sources, decoded_dim]


def build_role_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task, param_specs in ROLE_TASKS.items():
        _, r2_runs = load_cross_theta(task)
        final = r2_runs[:, -1]
        for dim, (symbol, meaning, role) in enumerate(param_specs):
            matched = final[:, dim, dim]
            off_mean = off_target_values(final, dim).mean(axis=1)
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS.get(task, task.replace("_", " ")),
                    "dim": dim,
                    "symbol": symbol,
                    "meaning": meaning,
                    "role": role,
                    "matched_mean": float(matched.mean()),
                    "matched_sd": float(matched.std()),
                    "off_mean": float(off_mean.mean()),
                    "off_sd": float(off_mean.std()),
                    "delta_mean": float((matched - off_mean).mean()),
                    "delta_sd": float((matched - off_mean).std()),
                }
            )
    return rows


def write_role_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_lines = [
        "task,dim,symbol,meaning,role,matched_mean,matched_sd,off_mean,off_sd,delta_mean,delta_sd"
    ]
    tex_rows: list[list[str]] = []
    midrules_after: list[int] = []
    last_task: str | None = None
    for row in rows:
        task = str(row["task"])
        if last_task is not None and task != last_task:
            midrules_after.append(len(tex_rows) - 1)
        tex_rows.append(
            [
                str(row["task_label"]),
                f"{row['symbol']} {row['meaning']}",
                str(row["role"]),
                fmt_pm(float(row["matched_mean"]), float(row["matched_sd"])),
                fmt_pm(float(row["off_mean"]), float(row["off_sd"])),
                fmt_pm(float(row["delta_mean"]), float(row["delta_sd"])),
            ]
        )
        csv_lines.append(
            ",".join(
                [
                    task,
                    str(row["dim"]),
                    str(row["symbol"]).replace(",", ""),
                    str(row["meaning"]).replace(",", ""),
                    str(row["role"]).replace(",", ""),
                    f"{row['matched_mean']:.6f}",
                    f"{row['matched_sd']:.6f}",
                    f"{row['off_mean']:.6f}",
                    f"{row['off_sd']:.6f}",
                    f"{row['delta_mean']:.6f}",
                    f"{row['delta_sd']:.6f}",
                ]
            )
        )
        last_task = task

    ROLE_CSV_OUT.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Wrote {ROLE_CSV_OUT}")
    write_tex_tabular(
        out_path=TABLE_DIR / "probe_parameter_role_summary.tex",
        columns=["Task", "Parameter", "Role", "Matched", "Off mean", "$\\Delta$"],
        rows=tex_rows,
        column_align="lllrrr",
        midrules_after=midrules_after,
        source_script="scripts/plot_probe_main_and_role_table.py",
    )


def main() -> None:
    plot_main_figure()
    rows = build_role_rows()
    write_role_table(rows)
    print("\nFinal-layer role table rows:")
    for row in rows:
        print(
            f"  {row['task_label']:<22} {row['symbol']} {row['meaning']:<18} "
            f"matched={row['matched_mean']:+.3f} off={row['off_mean']:+.3f}"
        )


if __name__ == "__main__":
    main()
