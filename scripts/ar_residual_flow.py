"""Conditional residual flow on top of TabPFN AR base distribution.

Phase-4 (B) — conditional version.

Pipeline:
  1. Simulate training data; extract TabPFN encoder embeddings (the
     standard context for our NSF flows).
  2. Fit `TabPFNAR` on (θ_train, x_train).
  3. For each training row x_i, draw a single AR sample θ_AR_i; compute
     residual r_i = θ_i − θ_AR_i. Same for val.
  4. Train a standard NSF modelling p(r | x, θ_AR_sample) — context is
     `concat(encoder(x), θ_AR_sample)`. This is the *conditional*
     residual; the flow learns to correct each specific AR guess.
  5. At inference per ref obs: draw θ_AR ~ TabPFNAR(·|x_ref, n=1000),
     condition the flow on `concat(encoder(x_ref), θ_AR)` (per-row
     context), draw r ~ NSF, output θ = θ_AR + r.

The unconditional version `p(r|x)` over-spreads samples because AR
and residual end up convolved: Var(θ) ≈ 2·Var(θ_AR) + Var(θ_true).
The conditional version learns θ_AR-specific corrections, so AR's
sampling noise is *cancelled* by the flow's correction rather than
added on top.

Saves samples in `flow_vs_quantile/{task}_s{seed}_residual_flow.npz`
matching the schema other diagnostics consume.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.ar_density import TabPFNAR  # noqa: E402
from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow, get_flow_defaults, sample_posterior, train_flow,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-flow-samples", type=int, default=1000)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "residual_flow"
    if args.n_train != 10000:
        suffix = f"{suffix}_n{args.n_train}"
    out_npz = OUT_DIR / f"{args.task}_s{args.seed}_{suffix}.npz"
    if out_npz.exists():
        print(f"[skip] {out_npz}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"task={args.task} seed={args.seed} (residual flow on AR base)")

    print("Simulating training data...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta, dim_x = data["dim_theta"], data["dim_x"]
    print(f"  dim_theta={dim_theta}, dim_x={dim_x}")

    task = get_task(args.task)
    prior_dist = task.get_prior_dist()

    # ── 1. Encoder embeddings (flow context) ──
    print("\nEmbedding train/val/ref at layer=None...")
    emb = TabPFNEmbedder(
        context_size=1000, seed=args.seed, label_strategy="per_dim",
        layer=None, model_type="regressor",
    )
    emb.fit(data["xs_train"], thetas=data["thetas_train"])
    emb_train = emb.transform(data["xs_train"]).astype(np.float32, copy=False)
    emb_val = emb.transform(data["xs_val"]).astype(np.float32, copy=False)

    x_ref = np.stack([
        task.get_observation(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    theta_true = np.stack([
        task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    e_ref = emb.transform(x_ref).astype(np.float32, copy=False)
    print(f"  emb_train {emb_train.shape}, e_ref {e_ref.shape}")

    # ── 2-3. AR base + residuals ──
    print("\nFitting TabPFNAR + drawing AR samples for train/val...")
    ar = TabPFNAR(prior=prior_dist, seed=args.seed)
    ar.fit(data["thetas_train"], data["xs_train"])

    theta_ar_train = ar.sample_batched(data["xs_train"]).cpu().numpy()
    theta_ar_val = ar.sample_batched(data["xs_val"]).cpu().numpy()
    r_train = (data["thetas_train"] - theta_ar_train).astype(np.float32)
    r_val = (data["thetas_val"] - theta_ar_val).astype(np.float32)
    print(f"  residual stats: train mean={r_train.mean():.4f} "
          f"std={r_train.std():.4f}; val mean={r_val.mean():.4f} "
          f"std={r_val.std():.4f}")

    # ── 4. Train NSF on conditional residuals p(r | x, θ_AR_sample) ──
    # Context = concat(encoder(x), θ_AR_sample). Each training row carries
    # the specific AR sample that produced its residual, so the flow
    # learns to *correct* that specific guess.
    ctx_train = np.concatenate(
        [emb_train, theta_ar_train.astype(np.float32)], axis=1,
    )
    ctx_val = np.concatenate(
        [emb_val, theta_ar_val.astype(np.float32)], axis=1,
    )
    print(f"  context shape: train {ctx_train.shape}, val {ctx_val.shape}")

    defaults = get_flow_defaults(dim_theta)
    print(f"\nBuilding NSF (n_transforms={defaults['n_transforms']}, "
          f"hidden={defaults['hidden_features']})...")
    flow = build_flow(
        dim_theta=dim_theta, dim_context=ctx_train.shape[1],
        n_transforms=defaults["n_transforms"],
        hidden_features=defaults["hidden_features"],
        n_bins=8,
    )
    cfg = TrainConfig(lr=args.lr, max_epochs=args.max_epochs)
    history = train_flow(flow, r_train, ctx_train, r_val, ctx_val, cfg)

    # ── 5. Inference per ref obs (conditional) ──
    # For each ref obs we need: (a) n_samples AR draws θ_AR; (b) one
    # residual sample per AR draw, conditioned on (encoder(x_ref),
    # θ_AR). zuko's flow(ctx).sample() with batched ctx draws one per row.
    print("\nSampling per ref obs...")
    flow_samples = np.zeros(
        (args.n_ref, args.n_flow_samples, dim_theta), dtype=np.float32,
    )
    flow.eval()
    device_ = next(flow.parameters()).device
    theta_mean_t = torch.tensor(
        history["theta_mean"], dtype=torch.float32, device=device_,
    )
    theta_std_t = torch.tensor(
        history["theta_std"], dtype=torch.float32, device=device_,
    )
    for i in range(args.n_ref):
        theta_ar_ref = ar.sample(
            x_obs=x_ref[i], n_samples=args.n_flow_samples,
        ).cpu().numpy()
        e_ref_rep = np.tile(e_ref[i], (args.n_flow_samples, 1))
        ctx_ref = np.concatenate(
            [e_ref_rep, theta_ar_ref.astype(np.float32)], axis=1,
        )
        ctx_ref_t = torch.tensor(ctx_ref, dtype=torch.float32, device=device_)
        with torch.no_grad():
            r_norm = flow(ctx_ref_t).sample()
            r_flow = (r_norm * theta_std_t + theta_mean_t).cpu().numpy()
        flow_samples[i] = (theta_ar_ref + r_flow).astype(np.float32)
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
