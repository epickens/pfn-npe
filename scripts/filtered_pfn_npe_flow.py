"""Run PFN-NPE NSF with query-local standardized-Euclidean filtering.

This is the P17 diagnostic for checking whether the AR method's top-k
context filter is itself providing unexpectedly strong signal at high
simulation budgets. For each sbibm reference observation, the script:

1. Simulates the full n_train pool.
2. Keeps the filter_context_size nearest simulations in standardized raw-x
   space, matching the AR filter criterion.
3. Trains the standard PFN-NPE NSF on that filtered local subset.
4. Samples only for that reference observation.

The output schema matches the other flow_vs_quantile npzs, so
scripts/run_c2st_sweep.py consumes it unchanged.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    sample_posterior,
    train_flow,
)
from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402
from scripts.layer_quantile_probe import _pinball_np  # noqa: E402

OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")


@dataclass
class TrainConfig:
    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 5.0
    flow_type: str = "nsf"
    hidden_features: list[int] = field(default_factory=lambda: [128, 128])


def _standardized_topk(
    xs_pool: np.ndarray,
    x_obs: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    k: int,
) -> np.ndarray:
    xs_z = (xs_pool - center) / scale
    x_z = (x_obs.reshape(1, -1) - center) / scale
    dists = np.linalg.norm(xs_z - x_z, axis=1)
    k = min(k, len(xs_pool))
    idx = np.argpartition(dists, kth=k - 1)[:k]
    return idx[np.argsort(dists[idx])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=100000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-flow-samples", type=int, default=1000)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--filter-context-size", type=int, default=10000)
    ap.add_argument("--context-size", type=int, default=1000,
                    help="TabPFN embedding context size inside each filtered subset.")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "nsf_filter"
    if args.n_train != 10000:
        suffix = f"{suffix}_n{args.n_train}"
    out_npz = OUT_DIR / f"{args.task}_s{args.seed}_{suffix}.npz"
    if out_npz.exists():
        print(f"[skip] {out_npz}")
        return
    partial_npz = OUT_DIR / "_partials" / out_npz.name
    partial_npz.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(
        f"task={args.task} seed={args.seed} n_train={args.n_train} "
        f"filter_context_size={args.filter_context_size}"
    )

    print("Simulating training data...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = data["dim_theta"]
    dim_x = data["dim_x"]
    print(f"  dim_theta={dim_theta}, dim_x={dim_x}")

    task = get_task(args.task)
    x_ref = np.stack([
        task.get_observation(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    theta_true = np.stack([
        task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    print(f"  x_ref {x_ref.shape}, theta_true {theta_true.shape}")

    x_center = data["xs_train"].mean(axis=0)
    x_scale = data["xs_train"].std(axis=0) + 1e-8

    defaults = get_flow_defaults(dim_theta)
    cfg = TrainConfig(lr=args.lr, max_epochs=args.max_epochs, flow_type="nsf")

    flow_samples_shape = (args.n_ref, args.n_flow_samples, dim_theta)
    train_indices_shape = (
        args.n_ref, min(args.filter_context_size, args.n_train),
    )
    flow_samples = np.zeros(flow_samples_shape, dtype=np.float32)
    train_indices = np.zeros(train_indices_shape, dtype=np.int64)
    done_obs = np.zeros(args.n_ref, dtype=bool)
    if partial_npz.exists():
        partial = np.load(partial_npz, allow_pickle=True)
        if (
            tuple(partial["flow_samples"].shape) == flow_samples_shape
            and tuple(partial["train_indices"].shape) == train_indices_shape
        ):
            flow_samples = partial["flow_samples"]
            train_indices = partial["train_indices"]
            done_obs = partial["done_obs"].astype(bool)
            print(f"[resume] loaded partial progress from {partial_npz}")

    for i in range(args.n_ref):
        if done_obs[i]:
            print(f"\n--- obs {i + 1}/{args.n_ref}: [skip] already complete ---")
            continue
        obs_start = time.perf_counter()
        idx_train = _standardized_topk(
            data["xs_train"], x_ref[i],
            center=x_center, scale=x_scale, k=args.filter_context_size,
        )
        idx_val = _standardized_topk(
            data["xs_val"], x_ref[i],
            center=x_center, scale=x_scale, k=min(args.n_val, args.filter_context_size),
        )
        train_indices[i] = idx_train

        xs_train_f = data["xs_train"][idx_train]
        thetas_train_f = data["thetas_train"][idx_train]
        xs_val_f = data["xs_val"][idx_val]
        thetas_val_f = data["thetas_val"][idx_val]

        print(
            f"\n--- obs {i + 1}/{args.n_ref}: "
            f"filtered train={len(idx_train)} val={len(idx_val)} ---"
        )
        emb = TabPFNEmbedder(
            context_size=args.context_size,
            seed=args.seed + i,
            label_strategy="per_dim",
            layer=None,
            model_type="regressor",
        )
        emb.fit(xs_train_f, thetas=thetas_train_f)
        emb_train = emb.transform(xs_train_f)
        emb_val = emb.transform(xs_val_f)
        e_obs = emb.transform(x_ref[i].reshape(1, -1))
        print(f"  emb_train {emb_train.shape}, e_obs {e_obs.shape}")

        flow = build_flow(
            dim_theta=dim_theta,
            dim_context=emb_train.shape[1],
            n_transforms=defaults["n_transforms"],
            hidden_features=defaults["hidden_features"],
            n_bins=8,
        )
        n_params = sum(p.numel() for p in flow.parameters())
        print(f"  Training NSF ({n_params} params)...")
        history = train_flow(
            flow,
            thetas_train_f,
            emb_train,
            thetas_val_f,
            emb_val,
            cfg,
        )
        flow_samples[i] = sample_posterior(
            flow,
            e_obs[0],
            history["theta_mean"],
            history["theta_std"],
            args.n_flow_samples,
        ).astype(np.float32)
        obs_end = time.perf_counter()
        print(
            f"  obs {i + 1}: drew {args.n_flow_samples} samples "
            f"in {obs_end - obs_start:.1f}s"
        )
        done_obs[i] = True
        np.savez(
            str(partial_npz),
            flow_samples=flow_samples,
            train_indices=train_indices,
            done_obs=done_obs,
        )

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
        flow_q=flow_q,
        emp_q=emp_q,
        flow_samples=flow_samples,
        theta_true=theta_true,
        x_ref=x_ref,
        flow_pinball=flow_pinball,
        task=args.task,
        seed=args.seed,
        flow_type=suffix,
        n_train_source=args.n_train,
        filter_context_size=args.filter_context_size,
        filter_type="standardized_euclidean",
        train_indices=train_indices,
    )
    partial_npz.unlink(missing_ok=True)
    print(f"\nflow pinball: {flow_pinball:.4f}")
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
