"""Oracle-repair C2ST plot for TabPFN-NPE PCA64 posterior samples.

The goal is to distinguish three failure modes:

* location/linear covariance calibration,
* marginal distribution calibration,
* residual rank/copula structure.

The repairs are post-hoc counterfactuals applied to saved posterior samples:

* mean+cov repair: affine-map model samples to reference mean/covariance,
* marginal repair: monotone quantile-map each model marginal to the reference,
* rank-only: existing pooled-rank C2ST from c2st_decomp, isolating dependence.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import compute_c2st, get_task  # noqa: E402


FLOW_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
C2ST_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation")
FIG_DIR = OUT_DIR / "figures"
METHOD = "nsf_pca64"
NAME_RE = re.compile(r"(?P<task>.+)_s(?P<seed>\d+)_(?P<method>.+)\.npz$")

TASK_ORDER = ["two_moons", "ar1_ts_t50", "slcp"]
TASK_LABEL = {
    "two_moons": "Two moons",
    "ar1_ts_t50": "AR(1), T=50",
    "slcp": "SLCP",
}
TASK_COLOR = {
    "two_moons": "#0072B2",
    "ar1_ts_t50": "#009E73",
    "slcp": "#D55E00",
}
CONDITION_ORDER = [
    "original",
    "mean_cov_repair",
    "marginal_repair",
    "rank_only",
]
CONDITION_LABEL = {
    "original": "Original",
    "mean_cov_repair": "Mean+cov\nmatched",
    "marginal_repair": "Marginals\nmatched",
    "rank_only": "Rank-only\nC2ST",
}
CONDITION_COLOR = {
    "original": "#4D4D4D",
    "mean_cov_repair": "#E69F00",
    "marginal_repair": "#56B4E9",
    "rank_only": "#CC79A7",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: tuple[str, ...]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def sym_matrix_power(cov: np.ndarray, power: float) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    floor = max(1e-8, 1e-6 * float(np.trace(cov)) / max(1, cov.shape[0]))
    vals = np.clip(vals, floor, None)
    return (vecs * (vals**power)) @ vecs.T


def mean_cov_repair(samples: np.ndarray, ref: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    model_mean = samples.mean(axis=0)
    ref_mean = ref.mean(axis=0)
    model_cov = np.atleast_2d(np.cov(samples, rowvar=False))
    ref_cov = np.atleast_2d(np.cov(ref, rowvar=False))
    model_invsqrt = sym_matrix_power(model_cov, -0.5)
    ref_sqrt = sym_matrix_power(ref_cov, 0.5)
    return (samples - model_mean) @ model_invsqrt @ ref_sqrt + ref_mean


def quantile_map_marginals(samples: np.ndarray, ref: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    mapped = np.empty_like(samples, dtype=np.float64)
    n = samples.shape[0]
    ref_n = ref.shape[0]
    for dim in range(samples.shape[1]):
        order = np.argsort(samples[:, dim], kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        q = (ranks + 0.5) / n

        ref_sorted = np.sort(ref[:, dim])
        pos = q * (ref_n - 1)
        lo = np.floor(pos).astype(int)
        hi = np.ceil(pos).astype(int)
        weight = pos - lo
        mapped[:, dim] = (1.0 - weight) * ref_sorted[lo] + weight * ref_sorted[hi]
    return mapped


def reference_samples(task_name: str, obs_idx: int, n: int) -> np.ndarray:
    ref = (
        get_task(task_name)
        .get_reference_posterior_samples(num_observation=obs_idx)
        .numpy()
        .astype(np.float64)
    )
    if len(ref) > n:
        rng = np.random.default_rng(137 + 31 * obs_idx)
        ref = ref[rng.choice(len(ref), size=n, replace=False)]
    return ref


def iter_flow_files(flow_dir: Path, method: str) -> list[tuple[str, int, Path, Path]]:
    out = []
    for path in sorted(flow_dir.glob(f"*_{method}.npz")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        task = match.group("task")
        seed = int(match.group("seed"))
        file_method = match.group("method")
        if task not in TASK_ORDER or file_method != method:
            continue
        c2st_path = C2ST_DIR / path.name
        if not c2st_path.exists():
            raise FileNotFoundError(f"Missing C2ST decomposition: {c2st_path}")
        out.append((task, seed, path, c2st_path))
    return out


def compute_rows(
    flow_dir: Path,
    method: str,
    *,
    n_ref: int,
    c2st_folds: int,
    c2st_max_epochs: int,
    c2st_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    files = iter_flow_files(flow_dir, method)
    if not files:
        raise FileNotFoundError(f"No flow files matched method {method!r} in {flow_dir}")

    for task, seed, flow_path, c2st_path in files:
        flow = np.load(flow_path)
        c2st = np.load(c2st_path)
        samples_all = flow["flow_samples"].astype(np.float64)
        n_obs = min(n_ref, samples_all.shape[0])
        print(f"[repair] {task} seed={seed} n_obs={n_obs}")
        for obs_zero in range(n_obs):
            obs = obs_zero + 1
            model = samples_all[obs_zero]
            ref = reference_samples(task, obs, len(model))
            n = min(len(model), len(ref))
            model = model[:n]
            ref = ref[:n]

            repaired_cov = mean_cov_repair(model, ref)
            repaired_marg = quantile_map_marginals(model, ref)
            mean_cov_c2st = compute_c2st(
                repaired_cov,
                ref,
                n_folds=c2st_folds,
                max_epochs=c2st_max_epochs,
                seed=c2st_seed,
            )
            marginal_c2st = compute_c2st(
                repaired_marg,
                ref,
                n_folds=c2st_folds,
                max_epochs=c2st_max_epochs,
                seed=c2st_seed,
            )
            rows.append({
                "task": task,
                "seed": seed,
                "obs": obs,
                "method": method,
                "original": float(c2st["joint"][obs_zero]),
                "mean_cov_repair": mean_cov_c2st,
                "marginal_repair": marginal_c2st,
                "rank_only": float(c2st["rank"][obs_zero]),
                "marginal_only": float(c2st["marginal"][obs_zero]),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_or_compute(args: argparse.Namespace) -> pd.DataFrame:
    csv_path = args.out_csv
    if csv_path.exists() and not args.force:
        print(f"Loading cached oracle repairs: {csv_path}")
        return pd.read_csv(csv_path)
    rows = compute_rows(
        args.flow_dir,
        args.method,
        n_ref=args.n_ref,
        c2st_folds=args.c2st_folds,
        c2st_max_epochs=args.c2st_max_epochs,
        c2st_seed=args.c2st_seed,
    )
    write_csv(csv_path, rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")
    return pd.DataFrame(rows)


def long_form(df: pd.DataFrame) -> pd.DataFrame:
    return df.melt(
        id_vars=["task", "seed", "obs", "method"],
        value_vars=CONDITION_ORDER,
        var_name="condition",
        value_name="c2st",
    )


def plot_oracle_repairs(df: pd.DataFrame, fig_dir: Path, formats: tuple[str, ...]) -> None:
    long = long_form(df)
    fig, ax = plt.subplots(figsize=(5.6, 3.7))

    x = np.arange(len(TASK_ORDER))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(CONDITION_ORDER))
    for idx, condition in enumerate(CONDITION_ORDER):
        means = [
            long.loc[
                (long["task"] == task) & (long["condition"] == condition),
                "c2st",
            ].mean()
            for task in TASK_ORDER
        ]
        sems = [
            long.loc[
                (long["task"] == task) & (long["condition"] == condition),
                "c2st",
            ].std(ddof=1)
            / np.sqrt((long["task"].eq(task) & long["condition"].eq(condition)).sum())
            for task in TASK_ORDER
        ]
        ax.bar(
            x + offsets[idx],
            means,
            width=width,
            yerr=sems,
            capsize=2,
            color=CONDITION_COLOR[condition],
            edgecolor="0.25",
            linewidth=0.5,
            label=CONDITION_LABEL[condition].replace("\n", " "),
        )
        for task_idx, task in enumerate(TASK_ORDER):
            vals = long.loc[
                (long["task"] == task) & (long["condition"] == condition),
                "c2st",
            ].to_numpy()
            jitter = np.linspace(-0.025, 0.025, len(vals))
            ax.scatter(
                np.full_like(vals, x[task_idx] + offsets[idx]) + jitter,
                vals,
                s=9,
                color=TASK_COLOR[task],
                alpha=0.35,
                edgecolor="none",
                zorder=3,
            )
    ax.axhline(0.5, color="0.45", ls=":", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Joint C2ST")
    ax.set_ylim(0.48, max(0.92, float(long["c2st"].max()) + 0.04))
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.tight_layout()
    save_figure(fig, fig_dir, "tabpfn_oracle_repair_c2st", formats)


def print_summary(df: pd.DataFrame) -> None:
    cols = ["original", "mean_cov_repair", "marginal_repair", "rank_only", "marginal_only"]
    summary = df.groupby("task")[cols].mean().loc[TASK_ORDER]
    print("\nMean C2ST by task")
    print(summary.round(3).to_string())
    print("\nFraction of original joint C2ST excess remaining")
    remain = pd.DataFrame(index=summary.index)
    denom = (summary["original"] - 0.5).clip(lower=1e-6)
    for col in ["mean_cov_repair", "marginal_repair", "rank_only"]:
        remain[col] = (summary[col] - 0.5) / denom
    print(remain.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-dir", type=Path, default=FLOW_DIR)
    ap.add_argument("--out-csv", type=Path, default=OUT_DIR / "tabpfn_npe_pca64_oracle_repairs.csv")
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--c2st-folds", type=int, default=5)
    ap.add_argument("--c2st-max-epochs", type=int, default=500)
    ap.add_argument("--c2st-seed", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    configure_style()
    df = load_or_compute(args)
    plot_oracle_repairs(df, args.fig_dir, tuple(args.formats))
    print_summary(df)
    print(f"\nWrote {args.fig_dir / 'tabpfn_oracle_repair_c2st.png'}")


if __name__ == "__main__":
    main()
