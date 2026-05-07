"""Validate the variance probe on a task with reference posterior samples.

Default: SLCP (sbibm), 10 reference observations with MCMC posterior samples.

For each layer k:
  1. Fit mean + variance probes on simulated train/val (same setup as
     layer_linear_probe.py).
  2. Extract embedding of each reference observation x_obs^(i).
  3. Predict sigma^2_probe(i, d) = exp(log_var_probe(emb(x_obs^(i)))).
  4. Compare to sigma^2_empirical(i, d) = Var of reference posterior samples
     for that observation.

Outputs:
  - Scatter of empirical vs predicted variance, colored by layer (or faceted).
  - Calibration summary per layer: Pearson r, slope, and log-space RMSE.
  - NLL of reference theta (if available) under predicted Gaussian.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402
from scripts.layer_linear_probe import (  # noqa: E402
    EPS, N_LAYERS, fit_mean_probe, fit_variance_probe,
)


def load_reference_data(task_name: str, n_ref: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_ref, emp_var_ref, theta_true_ref) stacked over the n_ref observations.

    emp_var is empirical posterior variance per θ-dim, computed from sbibm's
    reference posterior samples.
    """
    task = get_task(task_name)
    xs, vars_, thetas = [], [], []
    for i in range(1, n_ref + 1):
        x = task.get_observation(num_observation=i).numpy().reshape(-1)
        ref = task.get_reference_posterior_samples(num_observation=i).numpy()
        th_true = task.get_true_parameters(num_observation=i).numpy().reshape(-1)
        xs.append(x)
        vars_.append(ref.var(axis=0))
        thetas.append(th_true)
    return np.stack(xs), np.stack(vars_), np.stack(thetas)


def extract_or_load_cache(task: str, n_train: int, n_val: int, seed: int,
                          layer: int, cache: Path, data: dict):
    key = cache / f"{task}_n{n_train}_layer{layer}_s{seed}.npz"
    if key.exists():
        d = np.load(key)
        return d["emb_train"], d["emb_val"], d["thetas_train"], d["thetas_val"]
    cache.mkdir(parents=True, exist_ok=True)
    emb = TabPFNEmbedder(
        context_size=1000, seed=seed, label_strategy="per_dim",
        layer=layer, model_type="regressor",
    )
    emb.fit(data["xs_train"], thetas=data["thetas_train"])
    e_tr = emb.transform(data["xs_train"])
    e_va = emb.transform(data["xs_val"])
    np.savez(str(key),
             emb_train=e_tr, emb_val=e_va,
             thetas_train=data["thetas_train"], thetas_val=data["thetas_val"])
    return e_tr, e_va, data["thetas_train"], data["thetas_val"]


def embed_reference_obs(task: str, n_train: int, seed: int, layer: int,
                        data: dict, x_ref: np.ndarray) -> np.ndarray:
    """Extract embeddings of the n_ref reference observations at the given layer."""
    emb = TabPFNEmbedder(
        context_size=1000, seed=seed, label_strategy="per_dim",
        layer=layer, model_type="regressor",
    )
    emb.fit(data["xs_train"], thetas=data["thetas_train"])
    return emb.transform(x_ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-ref", type=int, default=10,
                    help="Reference observations (sbibm usually has 10)")
    ap.add_argument("--cache-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/probe/cache")
    ap.add_argument("--out-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/validate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    print(f"Simulating {args.task} (n_train={args.n_train}, n_val={args.n_val}, s={args.seed})...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = data["dim_theta"]
    print(f"  dim_theta={dim_theta}, dim_x={data['dim_x']}")

    print(f"Loading {args.n_ref} reference observations + posterior samples...")
    x_ref, emp_var, theta_true = load_reference_data(args.task, args.n_ref)
    print(f"  x_ref {x_ref.shape}, emp_var {emp_var.shape}, theta_true {theta_true.shape}")

    # Per-layer: fit probes, embed ref obs, predict sigma^2
    nll_val_by_layer = np.zeros(N_LAYERS)
    pred_var_by_layer = np.zeros((N_LAYERS, args.n_ref, dim_theta))
    pred_mu_by_layer = np.zeros_like(pred_var_by_layer)
    nll_ref_by_layer = np.zeros(N_LAYERS)  # NLL of true θ under predicted Gaussian

    for k in range(N_LAYERS):
        print(f"[layer {k}] fitting probes + embedding ref obs...")
        e_tr, e_va, th_tr, th_va = extract_or_load_cache(
            args.task, args.n_train, args.n_val, args.seed, k, cache, data,
        )
        mean_best = fit_mean_probe(e_tr, th_tr, e_va, th_va)
        var_best = fit_variance_probe(e_tr, th_tr, e_va, th_va,
                                      alpha_mu=mean_best["alpha"])
        nll_val_by_layer[k] = var_best["nll"]

        e_ref = embed_reference_obs(args.task, args.n_train, args.seed, k,
                                    data, x_ref)
        mu_ref = var_best["mu_model"].predict(e_ref)  # (n_ref, dim_theta)
        log_var_ref = var_best["log_var_model"].predict(e_ref)
        sigma_sq_ref = np.exp(log_var_ref) + EPS

        pred_mu_by_layer[k] = mu_ref
        pred_var_by_layer[k] = sigma_sq_ref

        # NLL of ground-truth θ under probe-predicted Gaussian
        resid_sq = (theta_true - mu_ref) ** 2
        nll_ref = 0.5 * (resid_sq / sigma_sq_ref
                         + np.log(sigma_sq_ref) + np.log(2 * np.pi))
        nll_ref_by_layer[k] = float(nll_ref.mean())

        # Per-layer calibration summary vs empirical posterior variance
        log_pred = np.log(sigma_sq_ref.reshape(-1))
        log_emp = np.log(emp_var.reshape(-1) + EPS)
        if np.std(log_pred) > 0 and np.std(log_emp) > 0:
            r_pearson, _ = pearsonr(log_pred, log_emp)
        else:
            r_pearson = float("nan")
        rmse_log = float(np.sqrt(np.mean((log_pred - log_emp) ** 2)))
        print(f"  val NLL={nll_val_by_layer[k]:+.4f}  "
              f"ref NLL={nll_ref_by_layer[k]:+.4f}  "
              f"log-r(pred,emp)={r_pearson:+.3f}  log-RMSE={rmse_log:.3f}")

    best_layer = int(np.argmin(nll_val_by_layer))
    print(f"\nBest val layer (by probe NLL): {best_layer}")

    # Save
    out_npz = out_dir / f"{args.task}_s{args.seed}.npz"
    np.savez(str(out_npz),
             layers=np.arange(N_LAYERS),
             nll_val=nll_val_by_layer,
             nll_ref=nll_ref_by_layer,
             pred_mu=pred_mu_by_layer,
             pred_var=pred_var_by_layer,
             emp_var=emp_var,
             theta_true=theta_true,
             x_ref=x_ref,
             task=args.task, seed=args.seed,
             best_layer=best_layer)
    print(f"Wrote {out_npz}")

    # ── Plot ────────────────────────────────────────────────────────────────
    # Panel A: NLL vs layer (val NLL and ref NLL on same axis, different series).
    # Panel B: scatter pred vs empirical variance at best layer, log-log.
    # Panel C: calibration per layer — log-space correlation + RMSE.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    layers = np.arange(N_LAYERS)
    axes[0].plot(layers, nll_val_by_layer, "o-", color="C0",
                 label="val NLL (2000 synthetic obs)")
    axes[0].plot(layers, nll_ref_by_layer, "s-", color="C3",
                 label=f"ref NLL ({args.n_ref} ref obs)")
    axes[0].axvline(best_layer, color="grey", ls=":", alpha=0.7)
    axes[0].set_xlabel("Encoder layer")
    axes[0].set_ylabel("Gaussian NLL")
    axes[0].set_title("A. NLL on val and reference obs")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    ax = axes[1]
    pv = pred_var_by_layer[best_layer].reshape(-1)
    ev = emp_var.reshape(-1)
    dim_labels = [f"θ{d}" for d in range(dim_theta)]
    for d in range(dim_theta):
        ax.loglog(emp_var[:, d], pred_var_by_layer[best_layer, :, d],
                  "o", ms=6, alpha=0.8, label=dim_labels[d])
    lo = min(pv.min(), ev.min()) * 0.5
    hi = max(pv.max(), ev.max()) * 2
    ax.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.7, label="y=x")
    ax.set_xlabel("Empirical posterior Var (from reference samples)")
    ax.set_ylabel("Probe-predicted σ²")
    ax.set_title(f"B. Calibration at best layer ({best_layer})")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    log_r = np.zeros(N_LAYERS)
    log_rmse = np.zeros(N_LAYERS)
    for k in range(N_LAYERS):
        lp = np.log(pred_var_by_layer[k].reshape(-1) + EPS)
        le = np.log(emp_var.reshape(-1) + EPS)
        if np.std(lp) > 0 and np.std(le) > 0:
            log_r[k], _ = pearsonr(lp, le)
        else:
            log_r[k] = float("nan")
        log_rmse[k] = float(np.sqrt(np.mean((lp - le) ** 2)))
    ax.plot(layers, log_r, "o-", color="C2", label="Pearson r (log-log)")
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel("log-Var Pearson r", color="C2")
    ax.tick_params(axis="y", labelcolor="C2")
    ax.axvline(best_layer, color="grey", ls=":", alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(layers, log_rmse, "s--", color="C1", label="log-Var RMSE")
    ax2.set_ylabel("log-Var RMSE", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax.set_title("C. Calibration quality per layer")

    fig.suptitle(f"Variance-probe validation: {args.task} | seed={args.seed} "
                 f"({args.n_ref} ref obs)")
    fig.tight_layout()
    out_png = out_dir / f"{args.task}_s{args.seed}.png"
    fig.savefig(str(out_png), dpi=140)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
