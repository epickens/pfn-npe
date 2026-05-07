"""PyMC NUTS (HMC) baseline for tasks with tractable likelihoods.

Runs PyMC NUTS for each of the 10 sbibm reference observations, drawing
1000 posterior samples per obs. No 'training' step: HMC pays the full cost
per query. The point of this baseline is to test whether classical MCMC
on a tractable likelihood can compete with NPE-PFN's per-query inference
in wall-clock terms.

Supported tasks (analytic / tractable likelihoods):
  - slcp                : x ~ N(μ(θ), Σ(θ)) i.i.d. ×4
  - slcp_distractors    : same as slcp + iid Gaussian noise dims (ignored)
  - gaussian_linear     : x = θ + ε,  θ ~ N(0,I), ε ~ N(0, σ² I)
  - gaussian_mixture    : x ~ 0.5·N(θ, σ_1² I) + 0.5·N(θ, σ_2² I)

Other tasks raise NotImplementedError. This is intentional --- we don't
want to silently fall back to a black-box likelihood (which would defeat
the "tractable HMC" comparison).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task  # noqa: E402
from scripts.layer_quantile_probe import _pinball_np  # noqa: E402

OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")


def build_slcp_model(x_obs: np.ndarray, distractor_dim: int = 0):
    """SLCP likelihood: 4 iid 2D Gaussian samples with θ-dependent params.

    x_obs is shape (8 + distractor_dim,). The first 8 dims are the SLCP
    observations (4 × 2D points); any extra dims are distractors and are
    ignored in the likelihood.
    """
    x_slcp = x_obs[:8].reshape(4, 2).astype(np.float64)
    with pm.Model() as model:
        theta = pm.Uniform("theta", lower=-3.0, upper=3.0, shape=5)
        mu = pt.stack([theta[0], theta[1]])
        s1 = theta[2] ** 2
        s2 = theta[3] ** 2
        rho = pt.tanh(theta[4])
        cov = pt.stack([
            pt.stack([s1 ** 2, rho * s1 * s2]),
            pt.stack([rho * s1 * s2, s2 ** 2]),
        ])
        # 4 iid 2D Gaussian observations
        for k in range(4):
            pm.MvNormal(f"x_{k}", mu=mu, cov=cov, observed=x_slcp[k])
    return model


def build_gaussian_linear_model(
    x_obs: np.ndarray,
    *,
    likelihood_std: float = 0.1,
    prior_std: float = 1.0,
    dim_theta: int | None = None,
):
    """gaussian_linear: x = θ + ε,  θ ~ N(0, I·prior_std²), ε ~ N(0, σ² I)."""
    D = x_obs.shape[0] if dim_theta is None else dim_theta
    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0.0, sigma=prior_std, shape=D)
        pm.Normal("x_obs", mu=theta, sigma=likelihood_std, observed=x_obs[:D])
    return model


def build_gaussian_mixture_model(x_obs: np.ndarray):
    """gaussian_mixture: x ~ 0.5 N(θ, 1) + 0.5 N(θ, 0.01),  θ ~ U(-10, 10)."""
    with pm.Model() as model:
        theta = pm.Uniform("theta", lower=-10.0, upper=10.0, shape=2)
        # log-mixture-of-Gaussians via potential
        x_t = pt.as_tensor_variable(x_obs.astype(np.float64))
        log_p1 = pm.logp(pm.Normal.dist(mu=theta, sigma=1.0), x_t).sum()
        log_p2 = pm.logp(pm.Normal.dist(mu=theta, sigma=0.1), x_t).sum()
        log_mix = pt.logaddexp(log_p1 + np.log(0.5), log_p2 + np.log(0.5))
        pm.Potential("mixture_lik", log_mix)
    return model


def build_model(task: str, x_obs: np.ndarray):
    if task == "slcp":
        return build_slcp_model(x_obs, distractor_dim=0)
    if task == "slcp_distractors":
        return build_slcp_model(x_obs, distractor_dim=x_obs.shape[0] - 8)
    if task == "gaussian_linear":
        return build_gaussian_linear_model(x_obs)
    if task == "gaussian_linear_uniform":
        return build_gaussian_linear_model(x_obs, prior_std=1.0)
    if task == "gaussian_mixture":
        return build_gaussian_mixture_model(x_obs)
    if task == "gaussian_mixture_distractors":
        return build_gaussian_mixture_model(x_obs[:2])
    raise NotImplementedError(
        f"PyMC HMC not implemented for task={task!r}. Tractable-likelihood tasks "
        f"only (slcp, slcp_distractors, gaussian_linear[_uniform], "
        f"gaussian_mixture[_distractors])."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10000,
                    help="Unused (kept for CLI uniformity with other baselines).")
    ap.add_argument("--n-val", type=int, default=2000,
                    help="Unused.")
    ap.add_argument("--n-flow-samples", type=int, default=1000)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--n-tune", type=int, default=500)
    ap.add_argument("--n-chains", type=int, default=2)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "pymc_hmc"
    out_npz = OUT_DIR / f"{args.task}_s{args.seed}_{suffix}.npz"
    if out_npz.exists():
        print(f"[skip] {out_npz}")
        return

    np.random.seed(args.seed)
    print(f"task={args.task} seed={args.seed}")

    task = get_task(args.task)
    x_ref = np.stack([
        task.get_observation(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    theta_true = np.stack([
        task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    dim_theta = theta_true.shape[1]

    # PyMC has no training stage; tag it as 0 for the wall-clock parser.
    print(f"[TIMING] phase=train duration=0.000")

    t_sample_start = time.perf_counter()
    print("\nSampling per ref obs (NUTS)...")
    flow_samples = np.zeros(
        (args.n_ref, args.n_flow_samples, dim_theta), dtype=np.float32,
    )
    draws_per_chain = max(1, args.n_flow_samples // args.n_chains)
    for i in range(args.n_ref):
        with build_model(args.task, x_ref[i]):
            idata = pm.sample(
                draws=draws_per_chain,
                tune=args.n_tune,
                chains=args.n_chains,
                random_seed=args.seed + i,
                progressbar=False,
                compute_convergence_checks=False,
            )
        # Pool chains, take first n_flow_samples.
        s = idata.posterior["theta"].values  # (chain, draw, D)
        s = s.reshape(-1, dim_theta)[: args.n_flow_samples]
        if s.shape[0] < args.n_flow_samples:
            pad = np.tile(s[-1:], (args.n_flow_samples - s.shape[0], 1))
            s = np.concatenate([s, pad], axis=0)
        flow_samples[i] = s.astype(np.float32)
        print(f"  obs {i+1}: drew {s.shape[0]} samples")
    t_sample_end = time.perf_counter()
    print(f"[TIMING] phase=sample duration={t_sample_end - t_sample_start:.3f}")

    taus = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float32)
    flow_q = np.zeros((args.n_ref, len(taus), dim_theta), dtype=np.float32)
    emp_q = np.zeros_like(flow_q)
    for i in range(args.n_ref):
        flow_q[i] = np.quantile(flow_samples[i], taus, axis=0)
        ref = task.get_reference_posterior_samples(num_observation=i + 1).numpy()
        emp_q[i] = np.quantile(ref, taus, axis=0)
    flow_pinball = float(_pinball_np(
        theta_true, flow_q.transpose(1, 0, 2), taus,
    ).mean())

    np.savez(
        str(out_npz),
        taus=taus, flow_q=flow_q, emp_q=emp_q,
        flow_samples=flow_samples,
        theta_true=theta_true, x_ref=x_ref,
        flow_pinball=flow_pinball,
        task=args.task, seed=args.seed, flow_type=suffix,
    )
    print(f"\nflow pinball: {flow_pinball:.4f}")
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
