"""Vanilla NPE baseline using sbi's SNPE_C with default density estimator.

Trains a standard MAF/NSF density estimator on (theta, x) pairs from the
sbibm task simulator (no PFN encoder), then samples 1000 draws for each of
the 10 reference observations. Same output schema as the other baselines
so downstream `c2st_decomposition` and `count_modes` can consume it.

Used in the wall-clock benchmark as a "no-PFN" point of comparison: shows
how much of PFN-NPE's amortized-sampling speed comes from the frozen TabPFN
encoder vs from the architecture in general.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from scripts.layer_quantile_probe import _pinball_np  # noqa: E402

OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-flow-samples", type=int, default=1000)
    ap.add_argument("--n-ref", type=int, default=10)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "vanilla_npe"
    if args.n_train != 10000:
        suffix = f"{suffix}_n{args.n_train}"
    out_npz = OUT_DIR / f"{args.task}_s{args.seed}_{suffix}.npz"
    if out_npz.exists():
        print(f"[skip] {out_npz}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"task={args.task} seed={args.seed} n_train={args.n_train}")

    t_train_start = time.perf_counter()
    print("Simulating training data...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = data["dim_theta"]
    dim_x = data["dim_x"]
    print(f"  dim_theta={dim_theta}, dim_x={dim_x}")

    task = get_task(args.task)
    prior_dist = task.get_prior_dist()

    from sbi.inference import SNPE_C
    from sbi.utils.user_input_checks import process_prior

    prior, *_ = process_prior(prior_dist)

    inference = SNPE_C(prior=prior, density_estimator="nsf")
    theta_t = torch.tensor(data["thetas_train"], dtype=torch.float32)
    x_t = torch.tensor(data["xs_train"], dtype=torch.float32)
    inference.append_simulations(theta_t, x_t)
    print(f"  attached {theta_t.shape[0]} simulations; training NSF...")

    posterior_estimator = inference.train(show_train_summary=False)
    posterior = inference.build_posterior(posterior_estimator)
    print(f"  trained NSF (default sbi config)")
    t_train_end = time.perf_counter()
    print(f"[TIMING] phase=train duration={t_train_end - t_train_start:.3f}")

    x_ref = np.stack([
        task.get_observation(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    theta_true = np.stack([
        task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])

    t_sample_start = time.perf_counter()
    print("\nSampling per ref obs...")
    flow_samples = np.zeros(
        (args.n_ref, args.n_flow_samples, dim_theta), dtype=np.float32,
    )
    for i in range(args.n_ref):
        x_obs_t = torch.tensor(x_ref[i], dtype=torch.float32)
        s = posterior.sample(
            (args.n_flow_samples,), x=x_obs_t, show_progress_bars=False,
        )
        flow_samples[i] = s.cpu().numpy().astype(np.float32)
        print(f"  obs {i+1}: drew {flow_samples.shape[1]} samples")
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
