"""Validate the quantile probe on a task with reference posterior samples.

Default: SLCP (sbibm), 10 reference observations with MCMC posterior samples.

For each layer k:
  1. Fit mean + quantile probes on simulated train/val (same setup as
     layer_quantile_probe.py).
  2. Extract embedding of each reference observation x_obs^(i).
  3. Predict q_τ(x_obs^(i)) for each τ in the probe's grid.
  4. Compare to empirical quantiles of the reference posterior samples.

Outputs:
  - Pinball loss of true θ on ref obs (probe-predicted quantiles).
  - τ-by-layer Pearson correlation heatmap between predicted and empirical
    quantiles, pooled across the n_ref observations and θ-dims.
  - Calibration of true θ vs predicted quantiles.
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
from scripts.layer_linear_probe import N_LAYERS, fit_mean_probe  # noqa: E402
from scripts.layer_quantile_probe import (  # noqa: E402
    DEFAULT_TAUS, _pinball_np, fit_quantile_probe,
)
from scripts.validate_variance_probe import extract_or_load_cache  # noqa: E402


def load_reference_quantiles(
    task_name: str, n_ref: int, taus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_ref, emp_q_ref, theta_true_ref).

    emp_q_ref shape: (n_ref, n_tau, dim_theta) — quantiles of reference
    posterior samples per observation.
    """
    task = get_task(task_name)
    xs, qs, thetas = [], [], []
    for i in range(1, n_ref + 1):
        x = task.get_observation(num_observation=i).numpy().reshape(-1)
        ref = task.get_reference_posterior_samples(num_observation=i).numpy()
        if hasattr(task, "get_true_parameters"):
            th_true = task.get_true_parameters(num_observation=i).numpy().reshape(-1)
        elif hasattr(task, "_theta_star"):
            # Some custom raw time-series tasks predate the sbibm-style
            # get_true_parameters method but store the same values internally.
            th_true = task._theta_star[i - 1].numpy().reshape(-1)
        else:
            raise AttributeError(
                f"{type(task).__name__} does not expose true parameters "
                "via get_true_parameters or _theta_star"
            )
        xs.append(x)
        qs.append(np.quantile(ref, taus, axis=0))           # (n_tau, dim_theta)
        thetas.append(th_true)
    return np.stack(xs), np.stack(qs), np.stack(thetas)


def embed_reference_obs(task: str, seed: int, layer: int,
                        data: dict, x_ref: np.ndarray) -> np.ndarray:
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
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--cache-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/probe/cache")
    ap.add_argument("--out-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/quantile_validate")
    ap.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    taus_arr = np.asarray(args.taus, dtype=np.float32)
    n_tau = len(taus_arr)

    print(f"Simulating {args.task} (n_train={args.n_train}, "
          f"n_val={args.n_val}, s={args.seed})...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = data["dim_theta"]
    print(f"  dim_theta={dim_theta}, dim_x={data['dim_x']}")

    print(f"Loading {args.n_ref} reference observations + posterior samples...")
    x_ref, emp_q, theta_true = load_reference_quantiles(
        args.task, args.n_ref, taus_arr,
    )
    print(f"  x_ref {x_ref.shape}, emp_q {emp_q.shape}, "
          f"theta_true {theta_true.shape}")

    pinball_val_by_layer = np.zeros(N_LAYERS)
    pinball_ref_by_layer = np.zeros(N_LAYERS)
    pred_q_by_layer = np.zeros((N_LAYERS, args.n_ref, n_tau, dim_theta))
    corr_by_layer = np.zeros((N_LAYERS, n_tau))
    rmse_by_layer = np.zeros((N_LAYERS, n_tau))

    for k in range(N_LAYERS):
        print(f"[layer {k}] fitting probes + embedding ref obs...")
        e_tr, e_va, th_tr, th_va = extract_or_load_cache(
            args.task, args.n_train, args.n_val, args.seed, k, cache, data,
        )
        mean_best = fit_mean_probe(e_tr, th_tr, e_va, th_va)
        q_best = fit_quantile_probe(
            e_tr, th_tr, e_va, th_va,
            alpha_mu=mean_best["alpha"], taus=tuple(args.taus),
        )
        pinball_val_by_layer[k] = q_best["pinball"]

        e_ref = embed_reference_obs(args.task, args.seed, k, data, x_ref)
        pq = q_best["model"].predict(e_ref)              # (n_ref, n_tau, dim_theta)
        pred_q_by_layer[k] = pq

        # Pinball of true θ under predicted quantiles
        pq_for_pinball = pq.transpose(1, 0, 2)            # (n_tau, n_ref, dim_theta)
        pinball_ref_by_layer[k] = float(
            _pinball_np(theta_true, pq_for_pinball, taus_arr).mean()
        )

        # Per-τ Pearson correlation pooled across (n_ref × dim_theta)
        for t_i in range(n_tau):
            pred = pq[:, t_i, :].reshape(-1)
            emp = emp_q[:, t_i, :].reshape(-1)
            if np.std(pred) > 0 and np.std(emp) > 0:
                corr_by_layer[k, t_i], _ = pearsonr(pred, emp)
            else:
                corr_by_layer[k, t_i] = float("nan")
            rmse_by_layer[k, t_i] = float(
                np.sqrt(np.mean((pred - emp) ** 2))
            )

        print(f"  val pinball={pinball_val_by_layer[k]:+.4f}  "
              f"ref pinball={pinball_ref_by_layer[k]:+.4f}  "
              f"corr_mean={np.nanmean(corr_by_layer[k]):+.3f}  "
              f"rmse_mean={np.mean(rmse_by_layer[k]):.3f}")

    best_layer = int(np.argmin(pinball_val_by_layer))
    print(f"\nBest val layer (by probe pinball): {best_layer}")

    out_npz = out_dir / f"{args.task}_s{args.seed}.npz"
    np.savez(
        str(out_npz),
        layers=np.arange(N_LAYERS), taus=taus_arr,
        pinball_val=pinball_val_by_layer,
        pinball_ref=pinball_ref_by_layer,
        pred_q=pred_q_by_layer.astype(np.float32),
        emp_q=emp_q.astype(np.float32),
        theta_true=theta_true,
        x_ref=x_ref,
        corr=corr_by_layer, rmse=rmse_by_layer,
        task=args.task, seed=args.seed, best_layer=best_layer,
    )
    print(f"Wrote {out_npz}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    layers = np.arange(N_LAYERS)

    ax = axes[0]
    ax.plot(layers, pinball_val_by_layer, "o-", color="C0",
            label=f"val pinball ({args.n_val} synthetic)")
    ax.plot(layers, pinball_ref_by_layer, "s-", color="C3",
            label=f"ref pinball ({args.n_ref} ref obs)")
    ax.axvline(best_layer, color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("Encoder layer"); ax.set_ylabel("Pinball loss")
    ax.set_title("A. Pinball on val and reference obs")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    pq_best = pred_q_by_layer[best_layer]                 # (n_ref, n_tau, dim_theta)
    cmap = plt.cm.viridis(np.linspace(0, 1, n_tau))
    for t_i, tau in enumerate(taus_arr):
        ax.scatter(
            emp_q[:, t_i, :].reshape(-1),
            pq_best[:, t_i, :].reshape(-1),
            color=cmap[t_i], s=40, edgecolor="black", lw=0.4, alpha=0.85,
            label=f"τ={tau}",
        )
    pmin = min(emp_q.min(), pq_best.min())
    pmax = max(emp_q.max(), pq_best.max())
    pad = 0.05 * (pmax - pmin)
    ax.plot([pmin - pad, pmax + pad], [pmin - pad, pmax + pad],
            "--", color="grey", alpha=0.7, label="y=x")
    ax.set_xlabel("Empirical q_τ (reference samples)")
    ax.set_ylabel("Probe-predicted q_τ")
    ax.set_title(f"B. Calibration at best layer ({best_layer})")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2]
    im = ax.imshow(
        corr_by_layer, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-1, vmax=1,
        extent=(-0.5, n_tau - 0.5, -0.5, N_LAYERS - 0.5),
    )
    ax.set_xticks(range(n_tau))
    ax.set_xticklabels([f"{t:.2f}" for t in taus_arr])
    ax.set_yticks(range(N_LAYERS))
    ax.set_xlabel("Quantile τ"); ax.set_ylabel("Encoder layer")
    ax.set_title("C. Pearson r (predicted vs empirical q_τ)")
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.axhline(best_layer, color="black", ls=":", alpha=0.6)

    fig.suptitle(f"Quantile-probe validation: {args.task} | seed={args.seed} "
                 f"({args.n_ref} ref obs)")
    fig.tight_layout()
    out_png = out_dir / f"{args.task}_s{args.seed}.png"
    fig.savefig(str(out_png), dpi=140)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
