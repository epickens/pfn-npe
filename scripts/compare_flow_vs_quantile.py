"""Head-to-head: trained flow's marginal quantiles vs linear-QR probe's.

For each (task, seed):
  1. Load default-config flow (`n10000_per_dim_regressor_s{seed}/flows/flow_tabpfn.pt`).
  2. Sample N=1000 θ samples from p_flow(θ | x_ref^i) for each of 10 sbibm
     reference observations.
  3. Compute marginal quantiles per θ-dim from the flow samples.
  4. Compare against:
       - Linear-QR probe predicted quantiles (from quantile_validate npz).
       - MCMC empirical quantiles (ground truth, from sbibm reference samples).
  5. Per-dim per-τ Pearson r against MCMC, plus pinball loss of true θ
     under each estimator's marginal CDF.

The motivating question: if QR recovers marginals well, is the flow ALSO
recovering them? If yes, the C2ST gap is in the joint (correlations or
multimodality). If the flow is worse than QR on marginals, the flow is
wasting decodable information and is the bottleneck.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow, get_flow_defaults, sample_posterior,
)
from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402
from scripts.layer_quantile_probe import _pinball_np  # noqa: E402

DEFAULT_TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
DEFAULT_LAYER = None  # match what `n10000_per_dim_regressor_s{seed}` was trained on


def load_flow(
    flow_path: Path, dim_theta: int, dim_context: int,
    n_bins: int = 8, flow_type: str = "nsf",
) -> torch.nn.Module:
    defaults = get_flow_defaults(dim_theta)
    flow = build_flow(
        dim_theta=dim_theta, dim_context=dim_context,
        n_transforms=defaults["n_transforms"],
        hidden_features=defaults["hidden_features"],
        n_bins=n_bins, flow_type=flow_type,
    )
    flow.load_state_dict(torch.load(flow_path, map_location="cpu", weights_only=True))
    flow.eval()
    if torch.cuda.is_available():
        flow = flow.cuda()
    return flow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-flow-samples", type=int, default=1000)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    ap.add_argument("--validate-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/quantile_validate")
    ap.add_argument("--out-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taus = np.asarray(args.taus, dtype=np.float32)
    n_tau = len(taus)

    flow_path = Path(
        f"pfn_testing/sbi/outputs/{args.task}/"
        f"n{args.n_train}_per_dim_regressor_s{args.seed}/flows/flow_tabpfn.pt"
    )
    if not flow_path.exists():
        raise FileNotFoundError(f"Trained flow not found: {flow_path}")

    qr_npz = Path(args.validate_dir) / f"{args.task}_s{args.seed}.npz"
    if not qr_npz.exists():
        raise FileNotFoundError(f"QR validate npz missing: {qr_npz}")
    qr = dict(np.load(qr_npz, allow_pickle=True))
    qr_best = int(qr["best_layer"])
    qr_pred = qr["pred_q"][qr_best]              # (n_ref, n_tau, dim_theta)
    emp_q = qr["emp_q"]                           # (n_ref, n_tau, dim_theta)
    theta_true = qr["theta_true"]                # (n_ref, dim_theta)
    x_ref = qr["x_ref"]                           # (n_ref, dim_x)

    print(f"task={args.task} seed={args.seed}")
    print(f"  loading {flow_path}")
    print(f"  QR best layer (per validate npz) = {qr_best}")

    print(f"Simulating training data for theta_mean/std + embedder context...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = data["dim_theta"]
    print(f"  dim_theta={dim_theta}, dim_x={data['dim_x']}")

    print("Embedding ref obs at layer=None (matches flow's training input)...")
    emb = TabPFNEmbedder(
        context_size=1000, seed=args.seed, label_strategy="per_dim",
        layer=DEFAULT_LAYER, model_type="regressor",
    )
    emb.fit(data["xs_train"], thetas=data["thetas_train"])
    emb_train = emb.transform(data["xs_train"])
    e_ref = emb.transform(x_ref)
    print(f"  emb_train {emb_train.shape}, e_ref {e_ref.shape}")

    theta_mean = data["thetas_train"].mean(axis=0)
    theta_std = data["thetas_train"].std(axis=0) + 1e-8
    print(f"  θ stats: mean={theta_mean}, std={theta_std}")

    flow = load_flow(flow_path, dim_theta=dim_theta, dim_context=e_ref.shape[1])
    print(f"  loaded flow ({sum(p.numel() for p in flow.parameters())} params)")

    flow_q = np.zeros((args.n_ref, n_tau, dim_theta), dtype=np.float32)
    flow_samples = np.zeros((args.n_ref, args.n_flow_samples, dim_theta),
                            dtype=np.float32)
    for i in range(args.n_ref):
        s = sample_posterior(
            flow, e_ref[i], theta_mean, theta_std, args.n_flow_samples,
        )
        flow_samples[i] = s
        flow_q[i] = np.quantile(s, taus, axis=0)

    # Per-dim per-τ correlations vs MCMC, RMSE in θ-space, and pinball of true θ
    pearson_flow = np.zeros((n_tau, dim_theta))
    pearson_qr = np.zeros((n_tau, dim_theta))
    rmse_flow = np.zeros((n_tau, dim_theta))
    rmse_qr = np.zeros((n_tau, dim_theta))
    for t_i in range(n_tau):
        for d in range(dim_theta):
            ef = flow_q[:, t_i, d]
            er = qr_pred[:, t_i, d]
            em = emp_q[:, t_i, d]
            if np.std(ef) > 0 and np.std(em) > 0:
                pearson_flow[t_i, d], _ = pearsonr(ef, em)
            else:
                pearson_flow[t_i, d] = float("nan")
            if np.std(er) > 0 and np.std(em) > 0:
                pearson_qr[t_i, d], _ = pearsonr(er, em)
            else:
                pearson_qr[t_i, d] = float("nan")
            rmse_flow[t_i, d] = float(np.sqrt(np.mean((ef - em) ** 2)))
            rmse_qr[t_i, d] = float(np.sqrt(np.mean((er - em) ** 2)))

    flow_pinball = float(_pinball_np(
        theta_true, flow_q.transpose(1, 0, 2), taus,
    ).mean())
    qr_pinball = float(_pinball_np(
        theta_true, qr_pred.transpose(1, 0, 2), taus,
    ).mean())

    # Per-dim summary
    print("\nPer-dim mean Pearson r vs MCMC quantiles:")
    print(f"{'dim':>3} {'flow_r_mean':>12} {'qr_r_mean':>12} "
          f"{'flow_rmse_mean':>15} {'qr_rmse_mean':>13}")
    for d in range(dim_theta):
        print(f"{d:>3} {np.nanmean(pearson_flow[:, d]):>+12.3f} "
              f"{np.nanmean(pearson_qr[:, d]):>+12.3f} "
              f"{np.mean(rmse_flow[:, d]):>15.4f} {np.mean(rmse_qr[:, d]):>13.4f}")

    print(f"\nPinball of true θ under flow marginal CDF: {flow_pinball:.4f}")
    print(f"Pinball of true θ under QR marginal CDF:    {qr_pinball:.4f}")

    out_npz = out_dir / f"{args.task}_s{args.seed}.npz"
    np.savez(
        str(out_npz),
        taus=taus,
        flow_q=flow_q, qr_q=qr_pred, emp_q=emp_q,
        flow_samples=flow_samples,
        theta_true=theta_true, x_ref=x_ref,
        pearson_flow=pearson_flow, pearson_qr=pearson_qr,
        rmse_flow=rmse_flow, rmse_qr=rmse_qr,
        flow_pinball=flow_pinball, qr_pinball=qr_pinball,
        task=args.task, seed=args.seed, qr_best_layer=qr_best,
    )
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
