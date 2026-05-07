"""Run our in-project autoregressive TabPFN density estimator on a benchmark.

Mirrors `scripts/npe_pfn_baseline.py` but uses `pfn_testing.sbi.ar_density.TabPFNAR`
instead of `npe_pfn.TabPFN_Based_NPE_PFN`. The output schema matches existing
`flow_vs_quantile/{task}_s{seed}_*.npz` so `c2st_decomposition.py` and
`count_modes.py` consume it unchanged.

Cross-task verification: pinball / joint / marg / rank C2ST should match
`{task}_s{seed}_npe_pfn.npz` results within ±0.02 (TabPFN RNG noise + lack of
rejection sampling).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.ar_density import TabPFNAR  # noqa: E402
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
    ap.add_argument("--filter-context-size", type=int, default=10000,
                    help="Cap on TabPFN's in-context rows. No-op when "
                         "n_train ≤ this value. Above the cap, the "
                         "filter selects rows per query (see "
                         "--filter-type).")
    ap.add_argument("--filter-type", default="standardized_euclidean",
                    choices=["standardized_euclidean", "random", "none"],
                    help="Per-query context selection above the cap. "
                         "'standardized_euclidean' (default) matches "
                         "NPE-PFN; 'random' is query-independent; "
                         "'none' disables filtering and lets TabPFN "
                         "subsample internally.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "our_ar"
    if args.n_train != 10000:
        suffix = f"{suffix}_n{args.n_train}"
    if args.filter_type != "standardized_euclidean":
        suffix = f"{suffix}_ft{args.filter_type}"
    if args.filter_context_size != 10000:
        suffix = f"{suffix}_fc{args.filter_context_size}"
    out_npz = OUT_DIR / f"{args.task}_s{args.seed}_{suffix}.npz"
    if out_npz.exists():
        print(f"[skip] {out_npz}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"task={args.task} seed={args.seed} n_train={args.n_train}")

    print("Simulating training data...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = data["dim_theta"]
    dim_x = data["dim_x"]
    print(f"  dim_theta={dim_theta}, dim_x={dim_x}")

    task = get_task(args.task)
    prior_dist = task.get_prior_dist()

    estimator = TabPFNAR(
        prior=prior_dist, seed=args.seed,
        filter_context_size=args.filter_context_size,
        filter_type=args.filter_type,
    )
    estimator.fit(data["thetas_train"], data["xs_train"])
    filt_active = (
        args.filter_type != "none"
        and estimator.n_train > args.filter_context_size
    )
    filt_str = (
        f"{args.filter_type}, ctx={args.filter_context_size}"
        if filt_active else "no-op"
    )
    print(f"  attached {estimator.n_train} simulations  "
          f"(filter: {filt_str})")

    x_ref = np.stack([
        task.get_observation(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    theta_true = np.stack([
        task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    print(f"  x_ref {x_ref.shape}, theta_true {theta_true.shape}")

    flow_samples = np.zeros(
        (args.n_ref, args.n_flow_samples, dim_theta), dtype=np.float32,
    )
    print("\nSampling per ref obs...")
    for i in range(args.n_ref):
        s = estimator.sample(
            x_obs=x_ref[i], n_samples=args.n_flow_samples,
        )
        flow_samples[i] = s.detach().cpu().numpy().astype(np.float32)
        print(f"  obs {i+1}: drew {flow_samples.shape[1]} samples")

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
        taus=taus,
        flow_q=flow_q, emp_q=emp_q,
        flow_samples=flow_samples,
        theta_true=theta_true, x_ref=x_ref,
        flow_pinball=flow_pinball,
        task=args.task, seed=args.seed, flow_type=suffix,
    )
    print(f"\nflow pinball: {flow_pinball:.4f}")
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
