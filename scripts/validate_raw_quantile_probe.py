"""Validate raw-observation quantile probes against reference posteriors.

This is the raw-x ablation for scripts/validate_quantile_probe.py. It fits the
same linear mean + multi-quantile heads directly on simulator observations x,
then evaluates predicted marginal posterior quantiles on reference
observations with MCMC reference posterior samples.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import simulate  # noqa: E402
from scripts.layer_linear_probe import fit_mean_probe  # noqa: E402
from scripts.layer_quantile_probe import (  # noqa: E402
    DEFAULT_TAUS, _pinball_np, fit_quantile_probe,
)
from scripts.validate_quantile_probe import load_reference_quantiles  # noqa: E402


def standardize_train_val_ref(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_val - mean) / std, (x_ref - mean) / std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--out-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/quantile_validate_raw")
    ap.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    ap.add_argument("--no-standardize", action="store_true",
                    help="Use raw x values directly. Default z-scores x using train statistics.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taus_arr = np.asarray(args.taus, dtype=np.float32)
    n_tau = len(taus_arr)

    print(f"Simulating {args.task} raw-x quantile probe "
          f"(n_train={args.n_train}, n_val={args.n_val}, s={args.seed})...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = data["dim_theta"]
    print(f"  dim_theta={dim_theta}, dim_x={data['dim_x']}")

    print(f"Loading {args.n_ref} reference observations + posterior samples...")
    x_ref, emp_q, theta_true = load_reference_quantiles(
        args.task, args.n_ref, taus_arr,
    )
    print(f"  x_ref {x_ref.shape}, emp_q {emp_q.shape}, "
          f"theta_true {theta_true.shape}")

    x_train = data["xs_train"].astype(np.float32)
    x_val = data["xs_val"].astype(np.float32)
    x_ref_probe = x_ref.astype(np.float32)
    if not args.no_standardize:
        x_train, x_val, x_ref_probe = standardize_train_val_ref(
            x_train, x_val, x_ref_probe,
        )
        print("  standardized x using training-set mean/std")

    print("Fitting raw-x mean + quantile probes...")
    mean_best = fit_mean_probe(x_train, data["thetas_train"], x_val, data["thetas_val"])
    q_best = fit_quantile_probe(
        x_train, data["thetas_train"],
        x_val, data["thetas_val"],
        alpha_mu=mean_best["alpha"], taus=tuple(args.taus),
    )
    pinball_val = float(q_best["pinball"])

    pred_q = q_best["model"].predict(x_ref_probe)       # (n_ref, n_tau, dim_theta)
    pred_for_pinball = pred_q.transpose(1, 0, 2)
    pinball_ref = float(_pinball_np(theta_true, pred_for_pinball, taus_arr).mean())

    corr = np.zeros(n_tau)
    rmse = np.zeros(n_tau)
    for t_i in range(n_tau):
        pred = pred_q[:, t_i, :].reshape(-1)
        emp = emp_q[:, t_i, :].reshape(-1)
        if np.std(pred) > 0 and np.std(emp) > 0:
            corr[t_i], _ = pearsonr(pred, emp)
        else:
            corr[t_i] = float("nan")
        rmse[t_i] = float(np.sqrt(np.mean((pred - emp) ** 2)))

    print(f"  val pinball={pinball_val:.4f}  "
          f"ref pinball={pinball_ref:.4f}  "
          f"corr_mean={np.nanmean(corr):+.3f}  rmse_mean={np.mean(rmse):.3f}")

    out_npz = out_dir / f"{args.task}_s{args.seed}.npz"
    np.savez(
        str(out_npz),
        layers=np.array([0]), taus=taus_arr,
        pinball_val=np.array([pinball_val]),
        pinball_ref=np.array([pinball_ref]),
        pred_q=pred_q[None].astype(np.float32),
        emp_q=emp_q.astype(np.float32),
        theta_true=theta_true,
        x_ref=x_ref,
        corr=corr[None],
        rmse=rmse[None],
        task=args.task, seed=args.seed, best_layer=0,
        representation="raw_x_standardized" if not args.no_standardize else "raw_x",
    )
    print(f"Wrote {out_npz}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar([0, 1], [pinball_val, pinball_ref], color=["C0", "C3"])
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["val", "ref"])
    axes[0].set_ylabel("Pinball loss")
    axes[0].set_title("A. Raw-x quantile pinball")
    axes[0].grid(True, axis="y", alpha=0.3)

    cmap = plt.cm.viridis(np.linspace(0, 1, n_tau))
    for t_i, tau in enumerate(taus_arr):
        axes[1].scatter(
            emp_q[:, t_i, :].reshape(-1),
            pred_q[:, t_i, :].reshape(-1),
            color=cmap[t_i], s=40, edgecolor="black", lw=0.4, alpha=0.85,
            label=f"tau={tau:.2f}",
        )
    pmin = min(emp_q.min(), pred_q.min())
    pmax = max(emp_q.max(), pred_q.max())
    pad = 0.05 * (pmax - pmin)
    axes[1].plot([pmin - pad, pmax + pad], [pmin - pad, pmax + pad],
                 "--", color="grey", alpha=0.7)
    axes[1].set_xlabel("Empirical q_tau")
    axes[1].set_ylabel("Raw-x predicted q_tau")
    axes[1].set_title("B. Reference quantile calibration")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(taus_arr, corr, "o-", color="C2", label="Pearson r")
    axes[2].set_xlabel("Quantile tau")
    axes[2].set_ylabel("Pearson r")
    axes[2].set_ylim(-0.1, 1.05)
    axes[2].set_title("C. Predicted vs empirical quantiles")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Raw-x quantile validation: {args.task} | seed={args.seed}")
    fig.tight_layout()
    out_png = out_dir / f"{args.task}_s{args.seed}.png"
    fig.savefig(str(out_png), dpi=140)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
