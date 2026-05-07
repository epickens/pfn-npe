"""Targeted local-SBI benchmark for PFN-as-proposal.

This benchmark compares methods in the observation-specific simulator-budget
regime. Each local method gets a global proposal stage plus local simulations
per observation. Runtime accounting is split into shared proposal fitting,
proposal sampling, local simulation, local NLE fitting/sampling, and metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sbi.inference import NLE, NPE
from sbi.neural_nets import likelihood_nn, posterior_nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfn_proposal_nle_refinement import (  # noqa: E402
    evaluate_method,
    get_uniform_prior_bounds,
    proposal_box,
    simulate_box_data,
    train_local_nle,
)
from pfn_testing.sbi.sbi_baselines import _prior_to_device  # noqa: E402
from pfn_testing.sbi.sbibm_utils import simulate  # noqa: E402
from pfn_weighted_fixed_table_nle import (  # noqa: E402
    PFNProposalModel,
    fit_pfn_proposal_model,
    sample_pfn_proposal,
)


@dataclass
class Proposal:
    method: str
    fit_seconds: float
    sample_fn: Callable[[np.ndarray, int], np.ndarray]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def train_npe_proposal(
    task: object,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    args: argparse.Namespace,
) -> Proposal:
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    torch.manual_seed(args.seed + 101)
    np.random.seed(args.seed + 101)
    prior = _prior_to_device(task.get_prior_dist(), device)
    inference = NPE(
        prior=prior,
        density_estimator=posterior_nn(model=args.npe_density_estimator),
        device=device,
    )
    inference.append_simulations(
        torch.tensor(theta_train, dtype=torch.float32),
        torch.tensor(x_train, dtype=torch.float32),
    )
    t0 = time.perf_counter()
    inference.train(
        training_batch_size=args.sbi_batch_size,
        stop_after_epochs=args.sbi_stop_after_epochs,
        max_num_epochs=args.sbi_max_num_epochs,
        learning_rate=args.sbi_lr,
    )
    posterior = inference.build_posterior()
    fit_seconds = time.perf_counter() - t0

    def sample_fn(obs_x: np.ndarray, n_samples: int) -> np.ndarray:
        samples = posterior.sample(
            (n_samples,),
            x=torch.tensor(obs_x, dtype=torch.float32, device=device),
        )
        return samples.detach().cpu().numpy().astype(np.float32)

    return Proposal("sbi_npe_proposal", fit_seconds, sample_fn)


def train_nle_proposal(
    task: object,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    args: argparse.Namespace,
) -> Proposal:
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    torch.manual_seed(args.seed + 202)
    np.random.seed(args.seed + 202)
    prior = _prior_to_device(task.get_prior_dist(), device)
    inference = NLE(
        prior=prior,
        density_estimator=likelihood_nn(model=args.nle_density_estimator),
        device=device,
    )
    inference.append_simulations(
        torch.tensor(theta_train, dtype=torch.float32),
        torch.tensor(x_train, dtype=torch.float32),
    )
    t0 = time.perf_counter()
    inference.train(
        training_batch_size=args.sbi_batch_size,
        stop_after_epochs=args.sbi_stop_after_epochs,
        max_num_epochs=args.sbi_max_num_epochs,
        learning_rate=args.sbi_lr,
    )
    posterior = inference.build_posterior(
        sample_with=args.nle_sample_with,
        mcmc_method=args.nle_mcmc_method,
    )
    fit_seconds = time.perf_counter() - t0

    def sample_fn(obs_x: np.ndarray, n_samples: int) -> np.ndarray:
        samples = posterior.sample(
            (n_samples,),
            x=torch.tensor(obs_x, dtype=torch.float32, device=device),
        )
        return samples.detach().cpu().numpy().astype(np.float32)

    return Proposal("sbi_nle_proposal", fit_seconds, sample_fn)


def train_pfn_proposal_shared(
    data: dict[str, Any],
    task_name: str,
    args: argparse.Namespace,
) -> Proposal:
    cfg_args = argparse.Namespace(**vars(args))
    cfg_args.task = task_name
    cfg_args.n_train = args.n_global_train
    t0 = time.perf_counter()
    model: PFNProposalModel = fit_pfn_proposal_model(
        {
            "thetas_train": data["thetas_train"][: args.n_global_train],
            "xs_train": data["xs_train"][: args.n_global_train],
            "thetas_val": data["thetas_val"],
            "xs_val": data["xs_val"],
            "task": data["task"],
            "dim_theta": data["dim_theta"],
            "dim_x": data["dim_x"],
        },
        cfg_args,
    )
    fit_seconds = time.perf_counter() - t0

    def sample_fn(obs_x: np.ndarray, n_samples: int) -> np.ndarray:
        return sample_pfn_proposal(model, obs_x, n_samples)

    return Proposal("pfn_proposal", fit_seconds, sample_fn)


def train_global_npe(
    task: object,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, Callable[[np.ndarray, int], np.ndarray]]:
    proposal = train_npe_proposal(task, theta_train, x_train, args)
    return proposal.fit_seconds, proposal.sample_fn


def train_global_nle(
    task: object,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, Callable[[np.ndarray, int], np.ndarray]]:
    proposal = train_nle_proposal(task, theta_train, x_train, args)
    return proposal.fit_seconds, proposal.sample_fn


def evaluate_samples(
    method: str,
    samples: np.ndarray,
    reference: np.ndarray,
    task_name: str,
    obs: int,
    seed: int,
    pair: tuple[int, int],
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    row = evaluate_method(method, samples, reference, pair, args)
    metric_seconds = time.perf_counter() - t0
    row.update(
        {
            "task": task_name,
            "obs": obs,
            "seed": seed,
            "metric_seconds": metric_seconds,
            **extra,
        }
    )
    return row


def run_task(task_name: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("=" * 92)
    print(f"Targeted local SBI benchmark: {task_name}")
    print("=" * 92)
    max_train = max(args.n_global_train, args.n_global_baseline_train)
    t0 = time.perf_counter()
    data = simulate(task_name, max_train, args.n_val, args.seed)
    global_sim_seconds = time.perf_counter() - t0
    task = data["task"]
    prior_low, prior_high = get_uniform_prior_bounds(task)
    obs_ids = list(range(1, args.n_observations + 1))
    pair = tuple(int(x.strip()) for x in args.pair.split(","))
    if len(pair) != 2:
        raise SystemExit("--pair must contain two comma-separated indices")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{task_name}_obs1-{args.n_observations}_"
        f"g{args.n_global_train}_l{args.n_local_train}_"
        f"gb{args.n_global_baseline_train}_ps{args.n_posterior_samples}_s{args.seed}"
    )
    detail_path = out_dir / f"{stem}_details.csv"

    detail_rows: list[dict[str, Any]] = []
    samples_payload: dict[str, np.ndarray] = {}

    proposals: list[Proposal] = []
    proposal_methods = [m.strip() for m in args.local_proposals.split(",") if m.strip()]
    if "pfn" in proposal_methods:
        proposals.append(train_pfn_proposal_shared(data, task_name, args))
    if "npe" in proposal_methods:
        proposals.append(
            train_npe_proposal(
                task,
                data["thetas_train"][: args.n_global_train],
                data["xs_train"][: args.n_global_train],
                args,
            )
        )
    if "nle" in proposal_methods:
        proposals.append(
            train_nle_proposal(
                task,
                data["thetas_train"][: args.n_global_train],
                data["xs_train"][: args.n_global_train],
                args,
            )
        )

    global_methods = [m.strip() for m in args.global_methods.split(",") if m.strip()]
    global_samplers: dict[str, tuple[float, Callable[[np.ndarray, int], np.ndarray]]] = {}
    if "npe" in global_methods:
        print(f"\n[global] Training sbi-NPE with {args.n_global_baseline_train} prior sims")
        global_samplers["global_sbi_npe"] = train_global_npe(
            task,
            data["thetas_train"][: args.n_global_baseline_train],
            data["xs_train"][: args.n_global_baseline_train],
            args,
        )
    if "nle" in global_methods:
        print(f"\n[global] Training sbi-NLE with {args.n_global_baseline_train} prior sims")
        global_samplers["global_sbi_nle"] = train_global_nle(
            task,
            data["thetas_train"][: args.n_global_baseline_train],
            data["xs_train"][: args.n_global_baseline_train],
            args,
        )

    for obs in obs_ids:
        print(f"\n[{task_name} obs {obs}]")
        obs_x = task.get_observation(num_observation=obs).numpy().squeeze(0)
        reference = task.get_reference_posterior_samples(num_observation=obs).numpy()
        reference = reference[: args.n_posterior_samples]
        samples_payload[f"obs{obs}_reference"] = reference
        samples_payload[f"obs{obs}_x"] = obs_x

        for method, (fit_seconds, sample_fn) in global_samplers.items():
            t_sample = time.perf_counter()
            samples = sample_fn(obs_x, args.n_posterior_samples)
            sample_seconds = time.perf_counter() - t_sample
            samples_payload[f"obs{obs}_{method}"] = samples
            row = evaluate_samples(
                method,
                samples,
                reference,
                task_name,
                obs,
                args.seed,
                pair,
                args,
                {
                    "regime": "global_prior",
                    "global_train_simulations": args.n_global_baseline_train,
                    "local_train_simulations": 0,
                    "total_simulations_per_observation": args.n_global_baseline_train,
                    "total_simulations_task": args.n_global_baseline_train,
                    "new_local_simulations": 0,
                    "shared_global_sim_seconds": global_sim_seconds,
                    "shared_fit_seconds": fit_seconds,
                    "proposal_sample_seconds": 0.0,
                    "posterior_sample_seconds": sample_seconds,
                    "local_sim_seconds": 0.0,
                    "local_nle_seconds": 0.0,
                    "box_volume_frac": np.nan,
                    "ref_box_coverage": np.nan,
                    "proposal_box_coverage": np.nan,
                    "total_method_seconds_obs": sample_seconds,
                },
            )
            detail_rows.append(row)
            write_csv(detail_path, detail_rows)
            print(
                f"  {method}: joint={row['joint_c2st']:.4f} "
                f"marg={row['marginal_c2st']:.4f} rank={row['rank_c2st']:.4f} "
                f"sample_s={sample_seconds:.1f}"
            )

        for proposal in proposals:
            t_prop = time.perf_counter()
            proposal_samples = proposal.sample_fn(obs_x, args.n_proposal_samples)
            proposal_sample_seconds = time.perf_counter() - t_prop
            box_low, box_high = proposal_box(
                proposal_samples,
                prior_low,
                prior_high,
                args.proposal_quantile_alpha,
                args.proposal_pad_frac,
                args.min_box_width_frac,
            )
            box_volume = float(np.prod((box_high - box_low) / (prior_high - prior_low)))
            ref_in_box = np.all((reference >= box_low) & (reference <= box_high), axis=1)
            prop_in_box = np.all((proposal_samples >= box_low) & (proposal_samples <= box_high), axis=1)

            t_local_sim = time.perf_counter()
            local_data = simulate_box_data(
                task,
                box_low,
                box_high,
                args.n_local_train,
                args.n_local_val,
                args.seed + 10_000 + 1_000 * obs,
                args.local_design,
                proposal_samples,
                args.jitter_scale,
            )
            local_sim_seconds = time.perf_counter() - t_local_sim

            t_local_nle = time.perf_counter()
            samples = train_local_nle(task, local_data, box_low, box_high, obs_x, args)
            local_nle_seconds = time.perf_counter() - t_local_nle
            method = f"{proposal.method}_local_nle"
            samples_payload[f"obs{obs}_{method}"] = samples
            row = evaluate_samples(
                method,
                samples,
                reference,
                task_name,
                obs,
                args.seed,
                pair,
                args,
                {
                    "regime": "local_targeted",
                    "global_train_simulations": args.n_global_train,
                    "local_train_simulations": args.n_local_train,
                    "total_simulations_per_observation": args.n_global_train + args.n_local_train,
                    "total_simulations_task": args.n_global_train + args.n_local_train * len(obs_ids),
                    "new_local_simulations": args.n_local_train,
                    "shared_global_sim_seconds": global_sim_seconds,
                    "shared_fit_seconds": proposal.fit_seconds,
                    "proposal_sample_seconds": proposal_sample_seconds,
                    "posterior_sample_seconds": 0.0,
                    "local_sim_seconds": local_sim_seconds,
                    "local_nle_seconds": local_nle_seconds,
                    "box_volume_frac": box_volume,
                    "ref_box_coverage": float(ref_in_box.mean()),
                    "proposal_box_coverage": float(prop_in_box.mean()),
                    "total_method_seconds_obs": (
                        proposal_sample_seconds + local_sim_seconds + local_nle_seconds
                    ),
                },
            )
            detail_rows.append(row)
            write_csv(detail_path, detail_rows)
            print(
                f"  {method}: joint={row['joint_c2st']:.4f} "
                f"marg={row['marginal_c2st']:.4f} rank={row['rank_c2st']:.4f} "
                f"cov={float(ref_in_box.mean()):.3f} "
                f"obs_s={row['total_method_seconds_obs']:.1f}"
            )

    summary_rows: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in detail_rows}):
        rows = [row for row in detail_rows if row["method"] == method]
        summary_rows.append(
            {
                "task": task_name,
                "method": method,
                "regime": rows[0]["regime"],
                "n_observations": len(rows),
                "joint_mean": mean_std([float(r["joint_c2st"]) for r in rows])[0],
                "joint_std_obs": mean_std([float(r["joint_c2st"]) for r in rows])[1],
                "marginal_mean": mean_std([float(r["marginal_c2st"]) for r in rows])[0],
                "marginal_std_obs": mean_std([float(r["marginal_c2st"]) for r in rows])[1],
                "rank_mean": mean_std([float(r["rank_c2st"]) for r in rows])[0],
                "rank_std_obs": mean_std([float(r["rank_c2st"]) for r in rows])[1],
                "pair_mean": mean_std([float(r["pair_copula_c2st"]) for r in rows])[0],
                "hist_tv_mean": mean_std([float(r["pair_hist_tv"]) for r in rows])[0],
                "total_simulations_per_observation": rows[0]["total_simulations_per_observation"],
                "total_simulations_task": rows[0]["total_simulations_task"],
                "shared_global_sim_seconds": rows[0]["shared_global_sim_seconds"],
                "shared_fit_seconds": rows[0]["shared_fit_seconds"],
                "sum_observation_seconds": float(sum(float(r["total_method_seconds_obs"]) for r in rows)),
                "mean_observation_seconds": mean_std([float(r["total_method_seconds_obs"]) for r in rows])[0],
                "mean_local_sim_seconds": mean_std([float(r["local_sim_seconds"]) for r in rows])[0],
                "mean_local_nle_seconds": mean_std([float(r["local_nle_seconds"]) for r in rows])[0],
                "mean_posterior_sample_seconds": mean_std(
                    [float(r["posterior_sample_seconds"]) for r in rows]
                )[0],
                "mean_ref_box_coverage": mean_std(
                    [
                        float(r["ref_box_coverage"])
                        for r in rows
                        if np.isfinite(float(r["ref_box_coverage"]))
                    ]
                    or [np.nan]
                )[0],
                "mean_box_volume_frac": mean_std(
                    [
                        float(r["box_volume_frac"])
                        for r in rows
                        if np.isfinite(float(r["box_volume_frac"]))
                    ]
                    or [np.nan]
                )[0],
            }
        )

    write_csv(out_dir / f"{stem}_details.csv", detail_rows)
    write_csv(out_dir / f"{stem}_summary.csv", summary_rows)
    with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "task": task_name,
                "summary": summary_rows,
                "details": detail_rows,
            },
            f,
            indent=2,
            default=str,
        )
    np.savez(str(out_dir / f"{stem}.npz"), **samples_payload)
    print("\nTask summary")
    print("-" * 92)
    for row in sorted(summary_rows, key=lambda r: r["joint_mean"]):
        print(
            f"{row['method']:<28} joint={row['joint_mean']:.4f} "
            f"marg={row['marginal_mean']:.4f} rank={row['rank_mean']:.4f} "
            f"sims/obs={row['total_simulations_per_observation']} "
            f"obs_s_mean={row['mean_observation_seconds']:.1f}"
        )
    return detail_rows, summary_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["two_moons", "slcp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-observations", type=int, default=10)
    ap.add_argument("--pair", default="0,1")
    ap.add_argument("--out-dir", default="pfn_testing/sbi/outputs/targeted_local_sbi_benchmark")

    ap.add_argument("--n-global-train", type=int, default=5_000)
    ap.add_argument("--n-local-train", type=int, default=10_000)
    ap.add_argument("--n-local-val", type=int, default=2_000)
    ap.add_argument("--n-global-baseline-train", type=int, default=15_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--n-posterior-samples", type=int, default=1_000)
    ap.add_argument("--n-proposal-samples", type=int, default=5_000)

    ap.add_argument("--local-proposals", default="pfn,npe,nle")
    ap.add_argument("--global-methods", default="npe,nle")
    ap.add_argument("--local-design", default="proposal_jitter", choices=["uniform_box", "proposal_jitter"])
    ap.add_argument("--jitter-scale", type=float, default=0.08)
    ap.add_argument("--proposal-quantile-alpha", type=float, default=0.05)
    ap.add_argument("--proposal-pad-frac", type=float, default=0.25)
    ap.add_argument("--min-box-width-frac", type=float, default=0.2)

    ap.add_argument("--context-size", type=int, default=1_000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--label-strategy", default="per_dim", choices=["per_dim", "per_dim_mean"])
    ap.add_argument("--model-type", default="regressor", choices=["regressor", "classifier"])
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"])
    ap.add_argument("--flow-type", default="nsf", choices=["nsf", "naf"])
    ap.add_argument("--n-transforms", type=int, default=None)
    ap.add_argument("--flow-epochs", type=int, default=100)
    ap.add_argument("--flow-patience", type=int, default=15)
    ap.add_argument("--flow-batch-size", type=int, default=256)
    ap.add_argument("--flow-lr", type=float, default=5e-4)

    ap.add_argument("--npe-density-estimator", default="maf")
    ap.add_argument("--nle-density-estimator", default="maf")
    ap.add_argument("--nle-sample-with", default="mcmc", choices=["mcmc"])
    ap.add_argument("--nle-mcmc-method", default="slice_np_vectorized")
    ap.add_argument("--nle-stop-after-epochs", type=int, default=20)
    ap.add_argument("--nle-max-num-epochs", type=int, default=200)
    ap.add_argument("--nle-batch-size", type=int, default=200)
    ap.add_argument("--nle-lr", type=float, default=5e-4)
    ap.add_argument("--sbi-stop-after-epochs", type=int, default=20)
    ap.add_argument("--sbi-max-num-epochs", type=int, default=200)
    ap.add_argument("--sbi-batch-size", type=int, default=200)
    ap.add_argument("--sbi-lr", type=float, default=5e-4)

    ap.add_argument("--c2st-folds", type=int, default=3)
    ap.add_argument("--c2st-max-epochs", type=int, default=100)
    ap.add_argument("--hist-bins", type=int, default=12)
    ap.add_argument("--conditional-bins", type=int, default=10)
    args = ap.parse_args()

    all_details: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for task_name in args.tasks:
        details, summaries = run_task(task_name, args)
        all_details.extend(details)
        all_summaries.extend(summaries)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_tag = "-".join(args.tasks)
    stem = (
        f"{task_tag}_obs1-{args.n_observations}_"
        f"g{args.n_global_train}_l{args.n_local_train}_"
        f"gb{args.n_global_baseline_train}_ps{args.n_posterior_samples}_s{args.seed}"
    )
    write_csv(out_dir / f"{stem}_all_details.csv", all_details)
    write_csv(out_dir / f"{stem}_all_summary.csv", all_summaries)
    print("\nCombined summary")
    print("-" * 92)
    for row in sorted(all_summaries, key=lambda r: (r["task"], r["joint_mean"])):
        print(
            f"{row['task']:<12} {row['method']:<28} "
            f"joint={row['joint_mean']:.4f} marg={row['marginal_mean']:.4f} "
            f"rank={row['rank_mean']:.4f} "
            f"obs_s_mean={row['mean_observation_seconds']:.1f}"
        )
    print(f"\nWrote {out_dir / f'{stem}_all_summary.csv'}")


if __name__ == "__main__":
    main()
