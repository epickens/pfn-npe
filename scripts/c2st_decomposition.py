"""Decompose flow vs MCMC distance into marginal and copula components.

For each (task, seed) where we have flow samples saved (from
flow_vs_quantile/), and per ref observation:

  joint_c2st        — vanilla C2ST on θ-space samples (the usual benchmark).
  marginal_c2st     — mean of 1D C2ST per θ-dim (marginal-only mismatch).
  rank_c2st         — C2ST on samples mapped to pooled-empirical-CDF ranks
                       (per-dim uniform by construction, so this isolates
                       copula / joint dependence mismatch).

If joint_c2st is much closer to rank_c2st than to marginal_c2st, the
remaining gap after marginals are matched lives in the copula — i.e.,
joint structure is the bottleneck.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import compute_c2st, get_task  # noqa: E402

CMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")


def pooled_rank_transform(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a, b into pooled empirical-CDF rank space per column.

    Both inputs share the same target marginal (uniform on (0, 1] per dim
    by construction). Differences post-transform are in the copula.
    """
    a_out = np.empty_like(a, dtype=np.float64)
    b_out = np.empty_like(b, dtype=np.float64)
    n_a = a.shape[0]
    n_b = b.shape[0]
    for d in range(a.shape[1]):
        pooled = np.concatenate([a[:, d], b[:, d]])
        order = np.argsort(pooled, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(pooled) + 1)
        ranks = ranks / (len(pooled) + 1)
        a_out[:, d] = ranks[:n_a]
        b_out[:, d] = ranks[n_a:n_a + n_b]
    return a_out, b_out


def per_obs_c2sts(flow_samples: np.ndarray, mcmc_samples: np.ndarray) -> dict:
    """Return joint, mean-marginal, and rank C2ST for a single observation."""
    n = min(len(flow_samples), len(mcmc_samples))
    a = flow_samples[:n]
    b = mcmc_samples[:n]
    joint = compute_c2st(a, b)

    marginals = [compute_c2st(a[:, d:d+1], b[:, d:d+1]) for d in range(a.shape[1])]
    marginal_mean = float(np.mean(marginals))

    a_r, b_r = pooled_rank_transform(a, b)
    rank = compute_c2st(a_r, b_r)
    return {
        "joint": joint,
        "marginal_mean": marginal_mean,
        "marginal_per_dim": np.asarray(marginals),
        "rank": rank,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--flow-type", default=None,
                    help="Tag for non-NSF estimators (e.g. 'fmpe', "
                         "'mixture_nsf_K2'). None = vanilla NSF (no suffix).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.flow_type}" if args.flow_type else ""
    p = CMP_DIR / f"{args.task}_s{args.seed}{suffix}.npz"
    if not p.exists():
        raise FileNotFoundError(f"flow_vs_quantile npz missing: {p}")
    cmp_data = dict(np.load(p, allow_pickle=True))
    flow_samples_all = cmp_data["flow_samples"]               # (n_ref, n_samples, dim_theta)

    task = get_task(args.task)
    print(f"task={args.task} seed={args.seed}")
    print(f"  flow_samples shape: {flow_samples_all.shape}")

    n_ref = min(args.n_ref, flow_samples_all.shape[0])
    dim_theta = flow_samples_all.shape[2]
    joint_arr = np.zeros(n_ref)
    marginal_arr = np.zeros(n_ref)
    marginal_per_dim_arr = np.zeros((n_ref, dim_theta))
    rank_arr = np.zeros(n_ref)

    for i in range(n_ref):
        mcmc = task.get_reference_posterior_samples(num_observation=i + 1).numpy()
        res = per_obs_c2sts(flow_samples_all[i], mcmc)
        joint_arr[i] = res["joint"]
        marginal_arr[i] = res["marginal_mean"]
        marginal_per_dim_arr[i] = res["marginal_per_dim"]
        rank_arr[i] = res["rank"]
        print(f"  obs {i+1}: joint={res['joint']:.3f}  "
              f"marg_mean={res['marginal_mean']:.3f}  rank={res['rank']:.3f}  "
              f"per_dim_marg={[f'{x:.3f}' for x in res['marginal_per_dim']]}")

    print(f"\nMean across {n_ref} obs:")
    print(f"  joint C2ST    : {joint_arr.mean():.4f} ± {joint_arr.std():.4f}")
    print(f"  marginal C2ST : {marginal_arr.mean():.4f} ± {marginal_arr.std():.4f}  "
          f"(per-dim mean)")
    print(f"  rank C2ST     : {rank_arr.mean():.4f} ± {rank_arr.std():.4f}")
    print(f"  joint − marg  : {(joint_arr - marginal_arr).mean():+.4f}  "
          f"(positive ⇒ joint has gap beyond marginals)")
    print(f"  rank − marg   : {(rank_arr - marginal_arr).mean():+.4f}  "
          f"(positive ⇒ copula contributes beyond marginals)")

    out_npz = OUT_DIR / f"{args.task}_s{args.seed}{suffix}.npz"
    np.savez(
        str(out_npz),
        joint=joint_arr, marginal=marginal_arr, rank=rank_arr,
        marginal_per_dim=marginal_per_dim_arr,
        task=args.task, seed=args.seed, flow_type=args.flow_type or "nsf",
    )
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
