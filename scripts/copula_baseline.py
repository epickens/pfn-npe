"""Run the Gaussian-copula posterior estimator on a benchmark task.

Wraps `pfn_testing.sbi.copula.GaussianCopulaSBI` with the same CLI and
output schema as `scripts/npe_pfn_baseline.py` so downstream
`c2st_decomposition.py` and `count_modes.py` can consume the result
unchanged.

Pipeline:
  1. Simulate train/val data via sbibm.
  2. Extract TabPFN encoder embeddings (per_dim regressor, layer=None).
  3. Fit `GaussianCopulaSBI`: trains a quantile probe on (emb, θ),
     uniformizes val θ via the probe, gaussianizes, and estimates Σ
     from val z-scores.
  4. For each of 10 sbibm reference observations, embed it and draw
     `n_flow_samples` posterior draws via the copula.
  5. Save samples + the fitted Σ to
     `flow_vs_quantile/{task}_s{seed}_copula.npz`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.copula import (  # noqa: E402
    DEFAULT_TAUS, FullyNeuralCopulaSBI, GaussianCopulaSBI, NeuralCopulaSBI,
)
from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402
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
    ap.add_argument("--n-taus", type=int, default=9,
                    help="Resolution of the τ-grid for marginal CDFs.")
    ap.add_argument("--copula-type", default="gaussian",
                    choices=["gaussian", "neural", "fully_neural"],
                    help="Copula model. 'gaussian' = constant Σ on quantile-"
                         "probe marginals; 'neural' = NSF on Gaussianized "
                         "residuals from quantile probe; 'fully_neural' = "
                         "1D NSF marginals + NSF copula on z.")
    ap.add_argument("--max-epochs", type=int, default=200,
                    help="(neural copula only) flow training epochs")
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="(neural copula only) flow learning rate")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = {
        "gaussian": "copula",
        "neural": "copula_neural",
        "fully_neural": "copula_fully_neural",
    }[args.copula_type]
    if args.n_train != 10000:
        suffix = f"{suffix}_n{args.n_train}"
    out_npz = OUT_DIR / f"{args.task}_s{args.seed}_{suffix}.npz"
    if out_npz.exists():
        print(f"[skip] {out_npz}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"task={args.task} seed={args.seed} (Gaussian copula on quantile marginals)")

    # ── 1. Simulate ──
    print("Simulating training data...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta, dim_x = data["dim_theta"], data["dim_x"]
    print(f"  dim_theta={dim_theta}, dim_x={dim_x}")

    task = get_task(args.task)
    prior_dist = task.get_prior_dist()

    # ── 2. Encoder embeddings ──
    print("\nEmbedding train/val/ref at layer=None...")
    embedder = TabPFNEmbedder(
        context_size=1000, seed=args.seed, label_strategy="per_dim",
        layer=None, model_type="regressor",
    )
    embedder.fit(data["xs_train"], thetas=data["thetas_train"])
    emb_train = embedder.transform(data["xs_train"]).astype(np.float32, copy=False)
    emb_val = embedder.transform(data["xs_val"]).astype(np.float32, copy=False)

    x_ref = np.stack([
        task.get_observation(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    theta_true = np.stack([
        task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    e_ref = embedder.transform(x_ref).astype(np.float32, copy=False)
    print(f"  emb_train {emb_train.shape}, e_ref {e_ref.shape}")

    # ── 3. Fit copula ──
    if args.n_taus == len(DEFAULT_TAUS):
        taus = DEFAULT_TAUS
    else:
        taus = tuple(np.linspace(0.025, 0.975, args.n_taus, dtype=np.float64).tolist())
    print(f"\nFitting {args.copula_type} copula...")
    if args.copula_type == "gaussian":
        copula = GaussianCopulaSBI(prior=prior_dist, seed=args.seed, taus=taus)
    elif args.copula_type == "neural":
        copula = NeuralCopulaSBI(
            prior=prior_dist, seed=args.seed, taus=taus,
            max_epochs=args.max_epochs, lr=args.lr,
        )
    else:  # fully_neural
        copula = FullyNeuralCopulaSBI(
            prior=prior_dist, seed=args.seed,
            max_epochs=args.max_epochs, lr=args.lr,
            marginal_max_epochs=args.max_epochs, marginal_lr=args.lr,
        )
    copula.fit(
        theta_train=data["thetas_train"], theta_val=data["thetas_val"],
        emb_train=emb_train, emb_val=emb_val,
    )

    corr = copula.correlation
    print(f"  Σ shape: {copula.sigma.shape}")
    print(f"  diag(Σ): {np.round(np.diag(copula.sigma), 3).tolist()}")
    if dim_theta <= 6:
        with np.printoptions(precision=3, suppress=True):
            print(f"  correlation matrix:\n{corr}")
    else:
        off_diag = corr[np.triu_indices(dim_theta, k=1)]
        print(f"  off-diag correlation summary: "
              f"mean={off_diag.mean():+.3f} std={off_diag.std():.3f} "
              f"max|r|={np.max(np.abs(off_diag)):.3f}")

    # ── 4. Sample for ref obs ──
    print("\nSampling per ref obs...")
    flow_samples = np.zeros(
        (args.n_ref, args.n_flow_samples, dim_theta), dtype=np.float32,
    )
    for i in range(args.n_ref):
        s = copula.sample(
            x_obs=x_ref[i], emb_obs=e_ref[i], n_samples=args.n_flow_samples,
        )
        flow_samples[i] = s.astype(np.float32)
        print(f"  obs {i+1}: drew {flow_samples.shape[1]} samples")

    # ── 5. Save in flow_vs_quantile schema ──
    taus_save = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float32)
    flow_q = np.zeros((args.n_ref, len(taus_save), dim_theta), dtype=np.float32)
    emp_q = np.zeros_like(flow_q)
    for i in range(args.n_ref):
        flow_q[i] = np.quantile(flow_samples[i], taus_save, axis=0)
        ref = task.get_reference_posterior_samples(num_observation=i + 1).numpy()
        emp_q[i] = np.quantile(ref, taus_save, axis=0)
    flow_pinball = float(_pinball_np(
        theta_true, flow_q.transpose(1, 0, 2), taus_save,
    ).mean())

    np.savez(
        str(out_npz),
        taus=taus_save,
        flow_q=flow_q, emp_q=emp_q,
        flow_samples=flow_samples,
        theta_true=theta_true, x_ref=x_ref,
        flow_pinball=flow_pinball,
        copula_sigma=copula.sigma,
        copula_taus=np.asarray(taus, dtype=np.float64),
        task=args.task, seed=args.seed, flow_type=suffix,
    )
    print(f"\nflow pinball: {flow_pinball:.4f}")
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
