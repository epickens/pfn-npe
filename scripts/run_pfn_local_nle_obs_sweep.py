"""Run PFN proposal + local NLE across observations and compare to baselines."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SBI_BASELINES = {
    "sbi-NPE": ("sbi_npe", "sbi_npe_joint_c2st", "sbi_npe_marginal_c2st", "sbi_npe_rank_c2st"),
    "sbi-NLE": ("sbi_nle", "sbi_nle_joint_c2st", "sbi_nle_marginal_c2st", "sbi_nle_rank_c2st"),
    "sbi-NRE": ("sbi_nre", "sbi_nre_joint_c2st", "sbi_nre_marginal_c2st", "sbi_nre_rank_c2st"),
    "sbi-FMPE": ("sbi_fmpe", "sbi_fmpe_joint_c2st", "sbi_fmpe_marginal_c2st", "sbi_fmpe_rank_c2st"),
}
CONTEXT_BASELINES = {
    "BayesFlow": ("bayesflow", "bayesflow_joint_c2st", "bayesflow_marginal_c2st", "bayesflow_rank_c2st"),
    "PFN-NPE PCA64": ("n10000_per_dim_regressor_pca_64", "tabpfn_joint_c2st", "tabpfn_marginal_c2st", "tabpfn_rank_c2st"),
    "Raw-x NSF": ("n10000_per_dim_regressor_pca_64", "raw_joint_c2st", "raw_marginal_c2st", "raw_rank_c2st"),
}


def obs_stem(args: argparse.Namespace, obs: int) -> str:
    return (
        f"{args.task}_obs{obs}_pfn{args.n_global_train}_"
        f"local{args.n_local_train}_{args.local_design}_"
        f"nle_ps{args.n_posterior_samples}_s{args.seed}"
    )


def run_obs(args: argparse.Namespace, obs: int, logs_dir: Path) -> Path:
    out_dir = Path(args.out_dir)
    summary = out_dir / f"{obs_stem(args, obs)}_summary.csv"
    if summary.exists() and not args.force:
        print(f"[obs {obs}] found existing {summary}")
        return summary

    log_path = logs_dir / f"{obs_stem(args, obs)}.log"
    cmd = [
        sys.executable,
        "-u",
        "scripts/pfn_local_nle_budget_test.py",
        "--task",
        args.task,
        "--obs",
        str(obs),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--n-global-train",
        str(args.n_global_train),
        "--n-val",
        str(args.n_val),
        "--skip-global-nle",
        "--context-size",
        str(args.context_size),
        "--embed-dim",
        str(args.embed_dim),
        "--n-proposal-samples",
        str(args.n_proposal_samples),
        "--n-local-train",
        str(args.n_local_train),
        "--n-local-val",
        str(args.n_local_val),
        "--local-design",
        args.local_design,
        "--n-posterior-samples",
        str(args.n_posterior_samples),
        "--c2st-folds",
        str(args.c2st_folds),
        "--c2st-max-epochs",
        str(args.c2st_max_epochs),
        "--out-dir",
        args.out_dir,
    ]
    print(f"[obs {obs}] running; log={log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            cmd,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(f"[obs {obs}] done")
    return summary


def read_method_row(path: Path, method: str) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == method:
                return row
    raise ValueError(f"Missing method {method!r} in {path}")


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def baseline_npz_path(task: str, tag: str, seed: int, n_train: int, ps: int) -> Path:
    base = Path("pfn_testing/sbi/outputs") / task
    if tag.startswith("sbi_") or tag == "bayesflow":
        suffix = f"{tag}_n{n_train}_ps{ps}"
        if seed != 42:
            suffix += f"_s{seed}"
        return base / suffix / "results" / "results.npz"
    suffix = f"{tag}_ps{ps}_s{seed}"
    return base / suffix / "results" / "results.npz"


def load_baselines(args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline_rows: list[dict[str, Any]] = []
    baselines = dict(SBI_BASELINES)
    if args.include_context_baselines:
        baselines.update(CONTEXT_BASELINES)
    for label, (tag, joint_key, marginal_key, rank_key) in baselines.items():
        path = baseline_npz_path(
            args.task,
            tag,
            args.seed,
            args.baseline_n_train,
            args.n_posterior_samples,
        )
        if not path.exists():
            print(f"[baseline] missing {label}: {path}")
            continue
        data = np.load(path, allow_pickle=True)
        baseline_rows.append(
            {
                "method": label,
                "training_simulations_total": args.baseline_n_train,
                "training_simulations_per_obs": args.baseline_n_train / len(args.obs),
                "joint_mean": float(np.mean(data[joint_key][np.asarray(args.obs) - 1])),
                "marginal_mean": float(np.mean(data[marginal_key][np.asarray(args.obs) - 1])),
                "rank_mean": float(np.mean(data[rank_key][np.asarray(args.obs) - 1])),
                "joint_std_obs": float(np.std(data[joint_key][np.asarray(args.obs) - 1], ddof=0)),
                "marginal_std_obs": float(np.std(data[marginal_key][np.asarray(args.obs) - 1], ddof=0)),
                "rank_std_obs": float(np.std(data[rank_key][np.asarray(args.obs) - 1], ddof=0)),
            }
        )
    return baseline_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_comparison(rows: list[dict[str, Any]]) -> None:
    print("\nSLCP Observation Sweep Comparison")
    print("-" * 92)
    print(
        f"{'method':<30} {'sims total':>11} {'sims/obs':>10} "
        f"{'joint':>8} {'marg':>8} {'rank':>8}"
    )
    for row in sorted(rows, key=lambda r: r["joint_mean"]):
        print(
            f"{row['method']:<30} "
            f"{row['training_simulations_total']:>11.0f} "
            f"{row['training_simulations_per_obs']:>10.0f} "
            f"{row['joint_mean']:>8.4f} "
            f"{row['marginal_mean']:>8.4f} "
            f"{row['rank_mean']:>8.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--obs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--n-global-train", type=int, default=5_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--context-size", type=int, default=1_000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--n-proposal-samples", type=int, default=5_000)
    ap.add_argument("--n-local-train", type=int, default=10_000)
    ap.add_argument("--n-local-val", type=int, default=2_000)
    ap.add_argument("--local-design", default="proposal_jitter")
    ap.add_argument("--n-posterior-samples", type=int, default=1_000)
    ap.add_argument("--c2st-folds", type=int, default=3)
    ap.add_argument("--c2st-max-epochs", type=int, default=100)

    ap.add_argument("--baseline-n-train", type=int, default=10_000)
    ap.add_argument("--include-context-baselines", action="store_true")
    ap.add_argument("--out-dir", default="pfn_testing/sbi/outputs/pfn_local_nle_budget_test")
    ap.add_argument("--logs-dir", default="results/logs/pfn_local_nle_obs_sweep")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    summaries = [run_obs(args, obs, logs_dir) for obs in args.obs]

    local_rows = []
    pfn_rows = []
    for obs, path in zip(args.obs, summaries, strict=True):
        local = read_method_row(path, f"pfn_local_nle_{args.local_design}")
        proposal = read_method_row(path, "pfn_proposal")
        local["obs"] = obs
        proposal["obs"] = obs
        local_rows.append(local)
        pfn_rows.append(proposal)

    detailed_path = (
        Path(args.out_dir)
        / (
            f"{args.task}_obs{'-'.join(str(o) for o in args.obs)}_"
            f"pfn{args.n_global_train}_local{args.n_local_train}_"
            f"{args.local_design}_ps{args.n_posterior_samples}_s{args.seed}_details.csv"
        )
    )
    write_csv(detailed_path, pfn_rows + local_rows)

    comparison_rows = []
    for label, rows, total_simulations in [
        ("PFN proposal", pfn_rows, args.n_global_train),
        (
            "PFN+local NLE",
            local_rows,
            args.n_global_train + args.n_local_train * len(args.obs),
        ),
    ]:
        comparison_rows.append(
            {
                "method": label,
                "training_simulations_total": total_simulations,
                "training_simulations_per_obs": total_simulations / len(args.obs),
                "joint_mean": mean_std([float(r["joint_c2st"]) for r in rows])[0],
                "marginal_mean": mean_std([float(r["marginal_c2st"]) for r in rows])[0],
                "rank_mean": mean_std([float(r["rank_c2st"]) for r in rows])[0],
                "pair_mean": mean_std([float(r["pair_copula_c2st"]) for r in rows])[0],
                "hist_tv_mean": mean_std([float(r["pair_hist_tv"]) for r in rows])[0],
                "ref_box_coverage_mean": mean_std([float(r["ref_box_coverage"]) for r in rows])[0],
                "box_volume_frac_mean": mean_std([float(r["box_volume_frac"]) for r in rows])[0],
            }
        )
    comparison_rows.extend(load_baselines(args))

    summary_path = (
        Path(args.out_dir)
        / (
            f"{args.task}_obs{'-'.join(str(o) for o in args.obs)}_"
            f"pfn{args.n_global_train}_local{args.n_local_train}_"
            f"{args.local_design}_vs_sbi_ps{args.n_posterior_samples}_s{args.seed}_summary.csv"
        )
    )
    json_path = summary_path.with_suffix(".json")
    write_csv(summary_path, comparison_rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": comparison_rows}, f, indent=2)

    print_comparison(comparison_rows)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {detailed_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
