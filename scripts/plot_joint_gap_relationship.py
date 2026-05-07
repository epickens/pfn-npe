"""Plot relationships among marginal/joint C2ST gaps and method gaps.

The main diagnostic question is whether tasks with large marginal-to-joint
errors are also the tasks where PFN-NPE loses to autoregressive NPE-PFN-style
sampling. This script uses existing c2st_decomp files; it does not run new
posterior sampling.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")
DEFAULT_BUDGET = 10_000

NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)(?:_(?P<rest>.+))?$")
N_SUFFIX_RE = re.compile(r"_n(\d+)$")

TASK_LABELS = {
    "two_moons": "Two moons",
    "gaussian_mixture": "Gauss. mix",
    "gaussian_linear": "Gauss. linear",
    "gaussian_linear_uniform": "Gauss. linear unif.",
    "bernoulli_glm": "Bern. GLM",
    "slcp": "SLCP",
    "sir": "SIR",
    "lotka_volterra": "Lotka-Volterra",
    "two_moons_distractors": "Two moons + distr.",
    "gaussian_mixture_distractors": "G. mix + distr.",
    "bernoulli_glm_distractors": "Bern. GLM + distr.",
    "sir_distractors": "SIR + distr.",
    "slcp_distractors": "SLCP + distr.",
    "ar1_ts_t50": "AR(1)",
    "ou": "OU",
    "solar_dynamo": "Solar dynamo",
}


def parse_name(stem: str) -> tuple[str, int, str, int] | None:
    match = NAME_RE.match(stem)
    if not match:
        return None
    task = match.group("task")
    seed = int(match.group("seed"))
    rest = match.group("rest") or ""
    n_match = N_SUFFIX_RE.search(rest)
    if n_match:
        budget = int(n_match.group(1))
        method = rest[: n_match.start()]
    else:
        budget = DEFAULT_BUDGET
        method = rest
    if not method:
        method = "nsf"
    return task, seed, method, budget


def load_rows(budget: int) -> dict[tuple[str, int, str], dict[str, float]]:
    rows: dict[tuple[str, int, str], dict[str, float]] = {}
    for path in sorted(DECOMP_DIR.glob("*.npz")):
        parsed = parse_name(path.stem)
        if not parsed:
            continue
        task, seed, method, path_budget = parsed
        if path_budget != budget:
            continue
        data = np.load(path, allow_pickle=True)
        joint = float(np.mean(data["joint"]))
        marginal = float(np.mean(data["marginal"]))
        rank = float(np.mean(data["rank"]))
        rows[(task, seed, method)] = {
            "joint": joint,
            "marginal": marginal,
            "rank": rank,
            "joint_minus_marginal": joint - marginal,
        }
    return rows


def matched_task_rows(
    data: dict[tuple[str, int, str], dict[str, float]],
    *,
    method_a: str,
    method_b: str,
) -> list[dict[str, float | str | int]]:
    by_task: dict[str, list[dict[str, float | str | int]]] = defaultdict(list)
    task_seed_pairs = sorted({(task, seed) for task, seed, _ in data})
    for task, seed in task_seed_pairs:
        a = data.get((task, seed, method_a))
        b = data.get((task, seed, method_b))
        if a is None or b is None:
            continue
        by_task[task].append(
            {
                "task": task,
                "seed": seed,
                "pfn_gap": a["joint_minus_marginal"],
                "ar_gap": b["joint_minus_marginal"],
                "pfn_joint": a["joint"],
                "ar_joint": b["joint"],
                "signed_method_gap": a["joint"] - b["joint"],
            }
        )

    out: list[dict[str, float | str | int]] = []
    for task, seed_rows in by_task.items():
        out.append(
            {
                "task": task,
                "seed": -1,
                "pfn_gap": float(np.mean([r["pfn_gap"] for r in seed_rows])),
                "ar_gap": float(np.mean([r["ar_gap"] for r in seed_rows])),
                "pfn_joint": float(np.mean([r["pfn_joint"] for r in seed_rows])),
                "ar_joint": float(np.mean([r["ar_joint"] for r in seed_rows])),
                "signed_method_gap": float(
                    np.mean([r["signed_method_gap"] for r in seed_rows])
                ),
                "n_seeds": len(seed_rows),
            }
        )
    return sorted(out, key=lambda r: str(r["task"]))


def corr_text(x: np.ndarray, y: np.ndarray) -> str:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return "r = n/a"
    r = float(np.corrcoef(x, y)[0, 1])
    return f"r = {r:.2f}"


def add_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 3 or np.std(x) == 0:
        return
    coef = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(xs, coef[0] * xs + coef[1], color="0.25", lw=1.2, ls="--")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument("--method-a", default="nsf", help="PFN-NPE method suffix")
    ap.add_argument("--method-b", default="npe_pfn", help="AR/NPE-PFN method suffix")
    ap.add_argument(
        "--out",
        type=Path,
        default=FIG_DIR / "joint_gap_relationship.png",
    )
    ap.add_argument(
        "--csv-out",
        type=Path,
        default=Path(
            "pfn_testing/sbi/outputs/layer_ablation/joint_gap_relationship.csv"
        ),
    )
    args = ap.parse_args()

    data = load_rows(args.budget)
    rows = matched_task_rows(data, method_a=args.method_a, method_b=args.method_b)
    if not rows:
        sys.exit("No matched task/seed rows found.")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pfn_gap = np.asarray([float(r["pfn_gap"]) for r in rows])
    ar_gap = np.asarray([float(r["ar_gap"]) for r in rows])
    signed_gap = np.asarray([float(r["signed_method_gap"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.scatter(pfn_gap, ar_gap, s=38, color="#0072B2", alpha=0.85)
    add_fit(ax, pfn_gap, ar_gap)
    ax.set_xlabel("PFN-NPE joint - marginal C2ST")
    ax.set_ylabel("NPE-PFN joint - marginal C2ST")
    ax.set_title(
        f"Do both methods struggle on the same tasks? ({corr_text(pfn_gap, ar_gap)})"
    )
    lim_max = max(float(pfn_gap.max()), float(ar_gap.max())) + 0.02
    lim_min = min(0.0, float(pfn_gap.min()), float(ar_gap.min())) - 0.01
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="0.7", lw=1.0, ls=":")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax = axes[1]
    ax.axhline(0.0, color="0.7", lw=1.0)
    ax.scatter(pfn_gap, signed_gap, s=38, color="#CC79A7", alpha=0.85)
    add_fit(ax, pfn_gap, signed_gap)
    ax.set_xlabel("PFN-NPE joint - marginal C2ST")
    ax.set_ylabel("PFN-NPE joint C2ST - NPE-PFN joint C2ST")
    ax.set_title(
        f"Does PFN joint gap predict relative loss? ({corr_text(pfn_gap, signed_gap)})"
    )

    for row in rows:
        label = TASK_LABELS.get(str(row["task"]), str(row["task"]).replace("_", " "))
        axes[0].annotate(
            label,
            (float(row["pfn_gap"]), float(row["ar_gap"])),
            fontsize=6,
            xytext=(3, 2),
            textcoords="offset points",
            alpha=0.75,
        )
        axes[1].annotate(
            label,
            (float(row["pfn_gap"]), float(row["signed_method_gap"])),
            fontsize=6,
            xytext=(3, 2),
            textcoords="offset points",
            alpha=0.75,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
