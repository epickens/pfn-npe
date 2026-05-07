"""Focused budget test for PFN proposal plus local NLE refinement.

This script asks the narrow question that matters for the manuscript: does a
PFN-derived proposal make a small local NLE refinement more useful than simply
training a global NLE with the same or larger simulation budget?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sbi.inference import NLE
from sbi.neural_nets import likelihood_nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfn_proposal_nle_refinement import (  # noqa: E402
    evaluate_method,
    get_uniform_prior_bounds,
    proposal_box,
    simulate_box_data,
    train_local_nle,
    train_pfn_proposal,
)
from pfn_testing.sbi.sbi_baselines import _prior_to_device  # noqa: E402
from pfn_testing.sbi.sbibm_utils import simulate  # noqa: E402


def subset_data(data: dict[str, Any], n_train: int) -> dict[str, Any]:
    return {
        "thetas_train": data["thetas_train"][:n_train],
        "xs_train": data["xs_train"][:n_train],
        "thetas_val": data["thetas_val"],
        "xs_val": data["xs_val"],
        "task": data["task"],
        "dim_theta": data["dim_theta"],
        "dim_x": data["dim_x"],
    }


def train_global_nle(
    task: object,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    obs_x: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)

    prior = _prior_to_device(task.get_prior_dist(), device)
    density_builder = likelihood_nn(model=args.nle_density_estimator)
    inference = NLE(prior=prior, density_estimator=density_builder, device=device)
    inference.append_simulations(
        torch.tensor(theta_train, dtype=torch.float32),
        torch.tensor(x_train, dtype=torch.float32),
    )
    inference.train(
        training_batch_size=args.nle_batch_size,
        stop_after_epochs=args.nle_stop_after_epochs,
        max_num_epochs=args.nle_max_num_epochs,
        learning_rate=args.nle_lr,
    )
    posterior = inference.build_posterior(
        sample_with=args.nle_sample_with,
        mcmc_method=args.nle_mcmc_method,
    )
    samples = posterior.sample(
        (args.n_posterior_samples,),
        x=torch.tensor(obs_x, dtype=torch.float32, device=device),
    )
    return samples.detach().cpu().numpy().astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\nFocused PFN Proposal + Local NLE Budget Test")
    print("-" * 116)
    print(
        f"{'method':<28} {'sims':>8} {'joint':>8} {'marg':>8} "
        f"{'rank':>8} {'pair':>8} {'histTV':>8} {'slope':>8}"
    )
    for row in rows:
        print(
            f"{row['method']:<28} "
            f"{row['total_simulations']:>8d} "
            f"{row['joint_c2st']:>8.4f} "
            f"{row['marginal_c2st']:>8.4f} "
            f"{row['rank_c2st']:>8.4f} "
            f"{row['pair_copula_c2st']:>8.4f} "
            f"{row['pair_hist_tv']:>8.4f} "
            f"{row['pair_conditional_slope_ratio']:>8.3f}"
        )


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--obs", type=int, default=1)
    ap.add_argument("--pair", default="0,1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--n-global-train", type=int, default=10_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--nle-train-sizes", default="5000,10000,15000")
    ap.add_argument(
        "--skip-global-nle",
        action="store_true",
        help="Only run PFN proposal/local NLE rows; useful when controls already exist.",
    )

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

    ap.add_argument("--n-proposal-samples", type=int, default=5_000)
    ap.add_argument("--proposal-quantile-alpha", type=float, default=0.05)
    ap.add_argument("--proposal-pad-frac", type=float, default=0.25)
    ap.add_argument("--min-box-width-frac", type=float, default=0.2)

    ap.add_argument("--n-local-train", type=int, default=5_000)
    ap.add_argument("--n-local-val", type=int, default=1_000)
    ap.add_argument("--local-design", default="proposal_jitter", choices=["uniform_box", "proposal_jitter"])
    ap.add_argument("--jitter-scale", type=float, default=0.08)

    ap.add_argument("--nle-density-estimator", default="maf")
    ap.add_argument("--nle-sample-with", default="mcmc", choices=["mcmc"])
    ap.add_argument("--nle-mcmc-method", default="slice_np_vectorized")
    ap.add_argument("--nle-stop-after-epochs", type=int, default=20)
    ap.add_argument("--nle-max-num-epochs", type=int, default=200)
    ap.add_argument("--nle-batch-size", type=int, default=200)
    ap.add_argument("--nle-lr", type=float, default=5e-4)

    ap.add_argument("--n-posterior-samples", type=int, default=1_000)
    ap.add_argument("--c2st-folds", type=int, default=3)
    ap.add_argument("--c2st-max-epochs", type=int, default=100)
    ap.add_argument("--hist-bins", type=int, default=12)
    ap.add_argument("--conditional-bins", type=int, default=10)
    ap.add_argument("--out-dir", default="pfn_testing/sbi/outputs/pfn_local_nle_budget_test")
    args = ap.parse_args()
    args.n_train = args.n_global_train

    nle_train_sizes = [] if args.skip_global_nle else parse_int_list(args.nle_train_sizes)
    if not args.skip_global_nle and not nle_train_sizes:
        raise SystemExit("--nle-train-sizes must contain at least one integer")
    pair_values = tuple(int(x.strip()) for x in args.pair.split(","))
    if len(pair_values) != 2:
        raise SystemExit("--pair must have two comma-separated indices")
    pair = (pair_values[0], pair_values[1])

    max_train = max([args.n_global_train, *nle_train_sizes])
    print(
        f"Simulating shared global data for {args.task}: "
        f"max_train={max_train}, n_val={args.n_val}, seed={args.seed}"
    )
    data = simulate(args.task, max_train, args.n_val, args.seed)
    task = data["task"]
    obs_x = task.get_observation(num_observation=args.obs).numpy().squeeze(0)
    reference = task.get_reference_posterior_samples(num_observation=args.obs).numpy()
    reference = reference[: args.n_posterior_samples]

    rows: list[dict[str, Any]] = []
    samples_to_save: dict[str, np.ndarray] = {
        "reference": reference,
        "obs_x": obs_x,
    }

    print("\n[PFN] Training proposal and local refinement")
    pfn_data = subset_data(data, args.n_global_train)
    t0 = time.perf_counter()
    proposal_samples = train_pfn_proposal(
        pfn_data,
        obs_x,
        args,
        args.n_proposal_samples,
    )
    proposal_seconds = time.perf_counter() - t0
    prior_low, prior_high = get_uniform_prior_bounds(task)
    box_low, box_high = proposal_box(
        proposal_samples,
        prior_low,
        prior_high,
        args.proposal_quantile_alpha,
        args.proposal_pad_frac,
        args.min_box_width_frac,
    )
    box_volume_frac = float(np.prod((box_high - box_low) / (prior_high - prior_low)))
    ref_in_box = np.all((reference >= box_low) & (reference <= box_high), axis=1)
    proposal_in_box = np.all((proposal_samples >= box_low) & (proposal_samples <= box_high), axis=1)
    print(f"  PFN box volume fraction: {box_volume_frac:.4f}")
    print(f"  Reference coverage: {ref_in_box.mean():.3f}")
    print(f"  Proposal coverage: {proposal_in_box.mean():.3f}")

    t0 = time.perf_counter()
    local_data = simulate_box_data(
        task,
        box_low,
        box_high,
        args.n_local_train,
        args.n_local_val,
        args.seed + 10_000,
        args.local_design,
        proposal_samples,
        args.jitter_scale,
    )
    local_nle_samples = train_local_nle(
        task,
        local_data,
        box_low,
        box_high,
        obs_x,
        args,
    )
    local_refinement_seconds = time.perf_counter() - t0

    pfn_eval_samples = proposal_samples[: args.n_posterior_samples]
    for method, samples, total_sims, local_sims, method_seconds, total_seconds in [
        (
            "pfn_proposal",
            pfn_eval_samples,
            args.n_global_train,
            0,
            proposal_seconds,
            proposal_seconds,
        ),
        (
            f"pfn_local_nle_{args.local_design}",
            local_nle_samples,
            args.n_global_train + args.n_local_train,
            args.n_local_train,
            local_refinement_seconds,
            proposal_seconds + local_refinement_seconds,
        ),
    ]:
        row = evaluate_method(method, samples, reference, pair, args)
        row.update(
            {
                "task": args.task,
                "obs": args.obs,
                "seed": args.seed,
                "global_train_simulations": args.n_global_train,
                "local_train_simulations": local_sims,
                "total_simulations": total_sims,
                "method_seconds": method_seconds,
                "total_method_seconds": total_seconds,
                "local_design": args.local_design,
                "box_volume_frac": box_volume_frac,
                "ref_box_coverage": float(ref_in_box.mean()),
                "proposal_box_coverage": float(proposal_in_box.mean()),
            }
        )
        rows.append(row)
        samples_to_save[method] = samples

    for n_train in nle_train_sizes:
        print(f"\n[NLE] Training global sbi-NLE with n_train={n_train}")
        t0 = time.perf_counter()
        samples = train_global_nle(
            task,
            data["thetas_train"][:n_train],
            data["xs_train"][:n_train],
            obs_x,
            args,
            seed_offset=n_train,
        )
        global_nle_seconds = time.perf_counter() - t0
        method = f"global_nle_n{n_train}"
        row = evaluate_method(method, samples, reference, pair, args)
        row.update(
            {
                "task": args.task,
                "obs": args.obs,
                "seed": args.seed,
                "global_train_simulations": n_train,
                "local_train_simulations": 0,
                "total_simulations": n_train,
                "method_seconds": global_nle_seconds,
                "total_method_seconds": global_nle_seconds,
                "local_design": "none",
                "box_volume_frac": np.nan,
                "ref_box_coverage": np.nan,
                "proposal_box_coverage": np.nan,
            }
        )
        rows.append(row)
        samples_to_save[method] = samples

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.task}_obs{args.obs}_pfn{args.n_global_train}_"
        f"local{args.n_local_train}_{args.local_design}_"
        f"nle{'-'.join(str(n) for n in nle_train_sizes)}_"
        f"ps{args.n_posterior_samples}_s{args.seed}"
    )
    csv_path = out_dir / f"{stem}_summary.csv"
    json_path = out_dir / f"{stem}.json"
    npz_path = out_dir / f"{stem}.npz"

    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": rows}, f, indent=2, default=str)
    np.savez(
        str(npz_path),
        box_low=box_low,
        box_high=box_high,
        **samples_to_save,
    )

    print_table(rows)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
