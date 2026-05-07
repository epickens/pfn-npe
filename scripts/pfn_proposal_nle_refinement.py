"""PFN proposal followed by observation-specific local NLE refinement.

The experiment tests whether PFN-NPE can be used as a cheap proposal/truncation
mechanism while NLE restores joint posterior structure inside the proposed
region. For a fixed observation:

1. Train the usual TabPFN embedding + conditional flow posterior estimator.
2. Draw PFN posterior samples and inflate their marginal quantile box.
3. Simulate local training data from a uniform prior over that box.
4. Train sbi-NLE on the local simulations.
5. Compare PFN proposal samples and local-NLE samples against the sbibm
   reference posterior, including a focused pairwise copula diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sbi.inference import NLE
from sbi.neural_nets import likelihood_nn
from sbi.utils import BoxUniform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    sample_posterior,
    train_flow,
)
from pfn_testing.sbi.sbibm_utils import compute_c2st, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import Config, PCAReducer, TabPFNEmbedder  # noqa: E402


def rank_columns(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    ranks = np.empty_like(samples, dtype=np.float64)
    denom = samples.shape[0] + 1.0
    for d in range(samples.shape[1]):
        order = np.argsort(samples[:, d], kind="mergesort")
        ranks[order, d] = (np.arange(samples.shape[0]) + 1.0) / denom
    return ranks


def pooled_ranks(samples: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(samples), len(reference))
    pooled = np.concatenate([samples[:n], reference[:n]], axis=0)
    ranks = rank_columns(pooled)
    return ranks[:n], ranks[n:]


def separate_pair_ranks(
    samples: np.ndarray,
    reference: np.ndarray,
    pair: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(samples), len(reference))
    return rank_columns(samples[:n, pair]), rank_columns(reference[:n, pair])


def hist2d(u: np.ndarray, bins: int) -> np.ndarray:
    h, _, _ = np.histogram2d(
        u[:, 0],
        u[:, 1],
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    h = h.astype(np.float64)
    return h / max(h.sum(), 1.0)


def conditional_slope_ratio(u_model: np.ndarray, u_ref: np.ndarray, n_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    model_means = []
    ref_means = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            m_mask = (u_model[:, 0] >= lo) & (u_model[:, 0] <= hi)
            r_mask = (u_ref[:, 0] >= lo) & (u_ref[:, 0] <= hi)
        else:
            m_mask = (u_model[:, 0] >= lo) & (u_model[:, 0] < hi)
            r_mask = (u_ref[:, 0] >= lo) & (u_ref[:, 0] < hi)
        model_means.append(float(u_model[m_mask, 1].mean()))
        ref_means.append(float(u_ref[r_mask, 1].mean()))
    x = np.arange(n_bins)
    ref_slope = np.polyfit(x, np.asarray(ref_means), 1)[0]
    model_slope = np.polyfit(x, np.asarray(model_means), 1)[0]
    return float(model_slope / ref_slope)


def get_uniform_prior_bounds(task: object) -> tuple[np.ndarray, np.ndarray]:
    prior_dist = task.get_prior_dist()
    if isinstance(prior_dist, torch.distributions.Independent) and isinstance(
        prior_dist.base_dist,
        torch.distributions.Uniform,
    ):
        low = prior_dist.base_dist.low.detach().cpu().numpy()
        high = prior_dist.base_dist.high.detach().cpu().numpy()
        return low.astype(np.float32), high.astype(np.float32)
    if isinstance(prior_dist, torch.distributions.Uniform):
        low = prior_dist.low.detach().cpu().numpy()
        high = prior_dist.high.detach().cpu().numpy()
        return low.astype(np.float32), high.astype(np.float32)
    raise TypeError(
        "Local NLE refinement currently supports box-uniform priors only. "
        f"Got {type(prior_dist).__name__}."
    )


def train_pfn_proposal(
    data: dict[str, Any],
    obs_x: np.ndarray,
    args: argparse.Namespace,
    n_samples: int,
) -> np.ndarray:
    print("\n[1/4] Training PFN-NPE proposal")
    embedder = TabPFNEmbedder(
        context_size=args.context_size,
        seed=args.seed,
        label_strategy=args.label_strategy,
        model_type=args.model_type,
        device=args.device,
        model_version=args.model_version,
    )
    embedder.fit(data["xs_train"], thetas=data["thetas_train"])
    emb_train = embedder.transform(data["xs_train"])
    emb_val = embedder.transform(data["xs_val"])
    emb_obs = embedder.transform(obs_x.reshape(1, -1))
    reducer = PCAReducer(args.embed_dim)
    ctx_train = reducer.fit_transform(emb_train)
    ctx_val = reducer.transform(emb_val)
    ctx_obs = reducer.transform(emb_obs)[0]

    dim_theta = data["dim_theta"]
    defaults = get_flow_defaults(dim_theta)
    cfg = Config(
        task_name=args.task,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        device=args.device,
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        label_strategy=args.label_strategy,
        model_type=args.model_type,
        flow_type=args.flow_type,
        max_epochs=args.flow_epochs,
        patience=args.flow_patience,
        batch_size=args.flow_batch_size,
        lr=args.flow_lr,
        n_posterior_samples=n_samples,
        n_reference_observations=1,
        skip_raw=True,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    flow = build_flow(
        dim_theta=dim_theta,
        dim_context=ctx_train.shape[1],
        n_transforms=args.n_transforms or defaults["n_transforms"],
        hidden_features=defaults["hidden_features"],
        n_bins=cfg.n_bins,
        flow_type=args.flow_type,
    )
    history = train_flow(
        flow,
        data["thetas_train"],
        ctx_train,
        data["thetas_val"],
        ctx_val,
        cfg,
    )
    samples = sample_posterior(
        flow,
        ctx_obs,
        history["theta_mean"],
        history["theta_std"],
        n_samples,
    )
    return samples.astype(np.float32)


def proposal_box(
    proposal_samples: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    quantile_alpha: float,
    pad_frac: float,
    min_width_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    in_prior = np.all(
        (proposal_samples >= prior_low) & (proposal_samples <= prior_high),
        axis=1,
    )
    if int(in_prior.sum()) >= max(100, 10 * proposal_samples.shape[1]):
        samples_for_box = proposal_samples[in_prior]
    else:
        samples_for_box = np.clip(proposal_samples, prior_low, prior_high)

    q_lo = np.quantile(samples_for_box, quantile_alpha, axis=0)
    q_hi = np.quantile(samples_for_box, 1.0 - quantile_alpha, axis=0)
    width = q_hi - q_lo
    lo = q_lo - pad_frac * width
    hi = q_hi + pad_frac * width

    min_width = min_width_frac * (prior_high - prior_low)
    center = 0.5 * (lo + hi)
    too_narrow = (hi - lo) < min_width
    lo[too_narrow] = center[too_narrow] - 0.5 * min_width[too_narrow]
    hi[too_narrow] = center[too_narrow] + 0.5 * min_width[too_narrow]

    lo = np.maximum(lo, prior_low)
    hi = np.minimum(hi, prior_high)
    return lo.astype(np.float32), hi.astype(np.float32)


def simulate_box_data(
    task: object,
    low: np.ndarray,
    high: np.ndarray,
    n_train: int,
    n_val: int,
    seed: int,
    design: str,
    proposal_samples: np.ndarray | None,
    jitter_scale: float,
) -> dict[str, np.ndarray]:
    print("\n[2/4] Simulating local training data in PFN box")
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val
    if design == "uniform_box":
        theta = rng.uniform(low=low, high=high, size=(n_total, len(low))).astype(np.float32)
    elif design == "proposal_jitter":
        if proposal_samples is None:
            raise ValueError("proposal_jitter requires proposal_samples")
        in_box = np.all((proposal_samples >= low) & (proposal_samples <= high), axis=1)
        base = proposal_samples[in_box]
        if len(base) < 50:
            base = np.clip(proposal_samples, low, high)
        chunks = []
        proposal_std = jitter_scale * (high - low)
        while sum(len(c) for c in chunks) < n_total:
            n_batch = max(n_total, 2_000)
            idx = rng.choice(len(base), size=n_batch, replace=True)
            cand = base[idx] + rng.normal(scale=proposal_std, size=(n_batch, len(low)))
            keep = np.all((cand >= low) & (cand <= high), axis=1)
            if np.any(keep):
                chunks.append(cand[keep].astype(np.float32))
        theta = np.concatenate(chunks, axis=0)[:n_total]
    else:
        raise ValueError(f"Unknown local design: {design}")
    simulator = task.get_simulator()
    x = simulator(torch.tensor(theta, dtype=torch.float32)).numpy().astype(np.float32)
    valid = np.isfinite(x).all(axis=1)
    if not np.all(valid):
        theta = theta[valid]
        x = x[valid]
        if len(theta) < n_total:
            raise RuntimeError(
                f"Only {len(theta)}/{n_total} local simulations were finite."
            )
        theta = theta[:n_total]
        x = x[:n_total]
    return {
        "theta_train": theta[:n_train],
        "x_train": x[:n_train],
        "theta_val": theta[n_train:],
        "x_val": x[n_train:],
    }


def train_local_nle(
    task: object,
    local_data: dict[str, np.ndarray],
    low: np.ndarray,
    high: np.ndarray,
    obs_x: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    print("\n[3/4] Training local sbi-NLE refinement")
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32, device=device),
        high=torch.tensor(high, dtype=torch.float32, device=device),
        device=device,
    )
    density_builder = likelihood_nn(model=args.nle_density_estimator)
    inference = NLE(prior=prior, density_estimator=density_builder, device=device)
    inference.append_simulations(
        torch.tensor(local_data["theta_train"], dtype=torch.float32),
        torch.tensor(local_data["x_train"], dtype=torch.float32),
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


def evaluate_method(
    method: str,
    samples: np.ndarray,
    reference: np.ndarray,
    pair: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    n = min(len(samples), len(reference), args.n_posterior_samples)
    model = samples[:n]
    ref = reference[:n]

    joint = compute_c2st(
        model,
        ref,
        n_folds=args.c2st_folds,
        max_epochs=args.c2st_max_epochs,
        seed=args.seed,
    )
    marginal_per_dim = []
    for d in range(model.shape[1]):
        marginal_per_dim.append(
            compute_c2st(
                model[:, [d]],
                ref[:, [d]],
                n_folds=args.c2st_folds,
                max_epochs=args.c2st_max_epochs,
                seed=args.seed + 100 + d,
            )
        )
    model_rank, ref_rank = pooled_ranks(model, ref)
    rank = compute_c2st(
        model_rank,
        ref_rank,
        n_folds=args.c2st_folds,
        max_epochs=args.c2st_max_epochs,
        seed=args.seed + 1_000,
    )
    u_model, u_ref = separate_pair_ranks(model, ref, pair)
    pair_copula = compute_c2st(
        u_model,
        u_ref,
        n_folds=args.c2st_folds,
        max_epochs=args.c2st_max_epochs,
        seed=args.seed + 2_000,
    )
    pair_hist_tv = float(0.5 * np.abs(hist2d(u_model, args.hist_bins) - hist2d(u_ref, args.hist_bins)).sum())
    return {
        "method": method,
        "joint_c2st": float(joint),
        "marginal_c2st": float(np.mean(marginal_per_dim)),
        "rank_c2st": float(rank),
        "pair_copula_c2st": float(pair_copula),
        "pair_hist_tv": pair_hist_tv,
        "pair_conditional_slope_ratio": conditional_slope_ratio(
            u_model,
            u_ref,
            args.conditional_bins,
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--obs", type=int, default=1)
    ap.add_argument("--pair", default="0,1")

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
    ap.add_argument("--local-design", default="uniform_box", choices=["uniform_box", "proposal_jitter"])
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
    ap.add_argument("--out-dir", default="pfn_testing/sbi/outputs/pfn_proposal_nle")
    args = ap.parse_args()

    pair_values = tuple(int(x.strip()) for x in args.pair.split(","))
    if len(pair_values) != 2:
        raise SystemExit("--pair must have two comma-separated indices")
    pair = (pair_values[0], pair_values[1])

    print(
        f"Simulating global data for {args.task}: "
        f"n_train={args.n_train}, n_val={args.n_val}, seed={args.seed}"
    )
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    task = data["task"]
    obs_x = task.get_observation(num_observation=args.obs).numpy().squeeze(0)
    reference = task.get_reference_posterior_samples(num_observation=args.obs).numpy()
    reference = reference[: args.n_posterior_samples]

    proposal_samples = train_pfn_proposal(data, obs_x, args, args.n_proposal_samples)
    prior_low, prior_high = get_uniform_prior_bounds(task)
    box_low, box_high = proposal_box(
        proposal_samples,
        prior_low,
        prior_high,
        args.proposal_quantile_alpha,
        args.proposal_pad_frac,
        args.min_box_width_frac,
    )
    box_width = box_high - box_low
    prior_width = prior_high - prior_low
    box_volume_frac = float(np.prod(box_width / prior_width))
    ref_in_box = np.all((reference >= box_low) & (reference <= box_high), axis=1)
    proposal_in_box = np.all(
        (proposal_samples >= box_low) & (proposal_samples <= box_high),
        axis=1,
    )
    print("\nPFN proposal box")
    print(f"  low={np.array2string(box_low, precision=3)}")
    print(f"  high={np.array2string(box_high, precision=3)}")
    print(f"  volume fraction of prior: {box_volume_frac:.4f}")
    print(f"  reference sample coverage: {ref_in_box.mean():.3f}")
    print(f"  proposal sample coverage: {proposal_in_box.mean():.3f}")

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
    local_nle_samples = train_local_nle(task, local_data, box_low, box_high, obs_x, args)

    print("\n[4/4] Evaluating proposal and refinement")
    pfn_eval_samples = proposal_samples[: args.n_posterior_samples]
    rows = [
        evaluate_method("pfn_proposal", pfn_eval_samples, reference, pair, args),
        evaluate_method("pfn_local_nle", local_nle_samples, reference, pair, args),
    ]
    for row in rows:
        row.update(
            {
                "task": args.task,
                "obs": args.obs,
                "n_train": args.n_train,
                "n_local_train": args.n_local_train,
                "local_design": args.local_design,
                "box_volume_frac": box_volume_frac,
                "ref_box_coverage": float(ref_in_box.mean()),
                "proposal_box_coverage": float(proposal_in_box.mean()),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.task}_obs{args.obs}_n{args.n_train}_{args.local_design}_"
        f"local{args.n_local_train}_"
        f"ps{args.n_posterior_samples}_s{args.seed}"
    )
    summary_csv = out_dir / f"{stem}_summary.csv"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    write_csv(summary_csv, rows)
    np.savez(
        str(npz_path),
        reference=reference,
        proposal_samples=proposal_samples,
        pfn_eval_samples=pfn_eval_samples,
        local_nle_samples=local_nle_samples,
        box_low=box_low,
        box_high=box_high,
        obs_x=obs_x,
    )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": rows}, f, indent=2, default=str)

    print("\nPFN Proposal + Local NLE Summary")
    print("-" * 104)
    print(
        f"{'method':<16} {'joint':>8} {'marg':>8} {'rank':>8} "
        f"{'pair':>8} {'histTV':>8} {'slope':>8}"
    )
    for row in rows:
        print(
            f"{row['method']:<16} "
            f"{row['joint_c2st']:>8.4f} "
            f"{row['marginal_c2st']:>8.4f} "
            f"{row['rank_c2st']:>8.4f} "
            f"{row['pair_copula_c2st']:>8.4f} "
            f"{row['pair_hist_tv']:>8.4f} "
            f"{row['pair_conditional_slope_ratio']:>8.3f}"
        )
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
