"""Quantile regression probes of TabPFN embeddings per layer.

For each of the 12 encoder layers:
 1. Load cached embeddings (same cache as layer_linear_probe.py).
 2. Fit a ridge mean probe to get the warm-start μ.
 3. Fit a joint multi-quantile probe: n_tau linear heads trained jointly to
    minimize pinball loss. α is swept; best α is picked on val pinball.
 4. Apply quantile rearrangement (sort predictions in τ at each (n, dim))
    so the reported quantiles are monotone by construction. The pre-sort
    crossing rate is retained as `crossing_rate` to flag underlying probe
    instability (low σ_homo + high-D embedding tends to drive τ heads
    together — see sir_distractors).

Writes a per-layer npz + a 4-panel summary png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.tabpfn_npe import simulate  # noqa: E402
from scripts.layer_linear_probe import (  # noqa: E402
    ALPHAS, EPS, N_LAYERS, extract_or_load, fit_mean_probe,
)

DEFAULT_TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)


class QuantileHead:
    """Multi-quantile linear head.

    W: (n_tau, D, dim_theta).  b: (n_tau, dim_theta).
    `predict(X)` returns shape (n, n_tau, dim_theta).

    By default predictions are rearranged (sorted along τ at each (n, dim))
    to enforce monotonicity. Quantile rearrangement (Chernozhukov, Fernández-
    Val & Galichon 2010) is provably non-increasing in pinball loss and
    eliminates τ-crossings produced by independent linear heads.
    """

    def __init__(self, W: np.ndarray, b: np.ndarray, taus: np.ndarray) -> None:
        self.W = W
        self.b = b
        self.taus = taus

    def predict(self, X: np.ndarray, rearrange: bool = True) -> np.ndarray:
        out = np.einsum("nd,tdk->ntk", X, self.W) + self.b[None]
        if rearrange:
            out = np.sort(out, axis=1)
        return out


def _pinball_np(th: np.ndarray, q: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """th (n, dim), q (n_tau, n, dim), taus (n_tau,) -> (n_tau, n, dim)."""
    r = th[None] - q
    return np.maximum(taus[:, None, None] * r, (taus[:, None, None] - 1.0) * r)


def fit_quantile_probe(
    e_tr: np.ndarray, th_tr: np.ndarray,
    e_va: np.ndarray, th_va: np.ndarray,
    alpha_mu: float = 1.0,
    alphas: tuple = ALPHAS,
    taus: tuple = DEFAULT_TAUS,
    n_epochs: int = 500,
    lr: float = 5e-3,
) -> dict:
    """Joint multi-τ linear quantile probe trained with pinball loss.

    Each τ head is warm-started from the ridge mean fit (W_τ = W_μ,
    b_τ = b_μ + Φ⁻¹(τ)·σ_homo). The α sweep is initialized to the
    constant-empirical-quantile baseline so the returned probe is
    fail-safe: if no α beats it, the constant baseline is reported.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_tr_t = torch.tensor(e_tr, dtype=torch.float32, device=device)
    th_tr_t = torch.tensor(th_tr, dtype=torch.float32, device=device)
    e_va_t = torch.tensor(e_va, dtype=torch.float32, device=device)
    th_va_t = torch.tensor(th_va, dtype=torch.float32, device=device)

    D = e_tr.shape[1]
    dim_theta = th_tr.shape[1]
    n_tr = e_tr.shape[0]
    taus_arr = np.asarray(taus, dtype=np.float32)
    n_tau = len(taus_arr)
    taus_t = torch.tensor(taus_arr, device=device).view(n_tau, 1, 1)

    mu_ridge = Ridge(alpha=alpha_mu).fit(e_tr, th_tr)
    W_mu_np = mu_ridge.coef_.T.astype(np.float32)
    b_mu_np = mu_ridge.intercept_.astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    mu_oof = np.zeros_like(th_tr)
    for tr_idx, ho_idx in kf.split(e_tr):
        m = Ridge(alpha=alpha_mu).fit(e_tr[tr_idx], th_tr[tr_idx])
        mu_oof[ho_idx] = m.predict(e_tr[ho_idx])
    sigma_homo = np.sqrt(((th_tr - mu_oof) ** 2).mean(axis=0) + EPS).astype(np.float32)

    q_emp = np.quantile(th_tr, taus_arr, axis=0).astype(np.float32)
    q_emp_pred_va = np.broadcast_to(
        q_emp[:, None, :], (n_tau, e_va.shape[0], dim_theta),
    ).copy()
    pinball_va_baseline = float(_pinball_np(th_va, q_emp_pred_va, taus_arr).mean())
    baseline_head = QuantileHead(
        np.zeros((n_tau, D, dim_theta), dtype=np.float32),
        q_emp.copy(),
        taus_arr,
    )

    baseline_pred = q_emp_pred_va.transpose(1, 0, 2).astype(np.float32)
    best: dict = {
        "alpha": np.inf,
        "pinball": pinball_va_baseline,
        "pinball_baseline": pinball_va_baseline,
        "model": baseline_head,
        "pred_quantiles_val": baseline_pred,
        "pred_quantiles_val_raw": baseline_pred.copy(),
    }

    z = norm.ppf(taus_arr).astype(np.float32)
    W_init = np.broadcast_to(W_mu_np[None], (n_tau, D, dim_theta)).copy()
    b_init = b_mu_np[None] + z[:, None] * sigma_homo[None]
    W_init_t = torch.tensor(W_init, device=device)
    b_init_t = torch.tensor(b_init, device=device)

    for a in alphas:
        W_q = torch.nn.Parameter(W_init_t.clone())
        b_q = torch.nn.Parameter(b_init_t.clone())
        opt = torch.optim.Adam([W_q, b_q], lr=lr)

        for _ in range(n_epochs):
            q_tr = torch.einsum("nd,tdk->tnk", e_tr_t, W_q) + b_q[:, None, :]
            r = th_tr_t.unsqueeze(0) - q_tr
            pinball = torch.maximum(taus_t * r, (taus_t - 1.0) * r)
            l2 = (a / n_tr) * (W_q ** 2).sum()
            loss = pinball.mean() + l2
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            q_va = torch.einsum("nd,tdk->tnk", e_va_t, W_q) + b_q[:, None, :]
            q_va_raw = q_va.cpu().numpy().transpose(1, 0, 2)
            q_va_sorted, _ = torch.sort(q_va, dim=0)
            r_va = th_va_t.unsqueeze(0) - q_va_sorted
            pinball_va = torch.maximum(taus_t * r_va, (taus_t - 1.0) * r_va)
            val_pinball = float(pinball_va.mean())
            if val_pinball < best["pinball"]:
                W_np = W_q.detach().cpu().numpy()
                b_np = b_q.detach().cpu().numpy()
                head = QuantileHead(W_np, b_np, taus_arr)
                best = {
                    "alpha": a,
                    "pinball": val_pinball,
                    "pinball_baseline": pinball_va_baseline,
                    "model": head,
                    "pred_quantiles_val": q_va_sorted.cpu().numpy().transpose(1, 0, 2),
                    "pred_quantiles_val_raw": q_va_raw,
                }

    pq = best["pred_quantiles_val"]
    pinball_full = _pinball_np(th_va, pq.transpose(1, 0, 2), taus_arr)
    pinball_per_tau = pinball_full.mean(axis=(1, 2))
    pinball_per_dim = pinball_full.mean(axis=(0, 1))

    pq_raw = best["pred_quantiles_val_raw"]
    cross = 0
    total = 0
    for i in range(n_tau):
        for j in range(i + 1, n_tau):
            cross += int((pq_raw[:, i, :] > pq_raw[:, j, :]).sum())
            total += pq_raw.shape[0] * pq_raw.shape[2]
    crossing_rate = cross / max(total, 1)

    cov = (th_va[:, None, :] <= pq).mean(axis=(0, 2))

    if 0.25 in taus_arr and 0.75 in taus_arr:
        i25 = int(np.where(taus_arr == 0.25)[0][0])
        i75 = int(np.where(taus_arr == 0.75)[0][0])
        iqr_val = pq[:, i75, :] - pq[:, i25, :]
    else:
        iqr_val = np.zeros((pq.shape[0], dim_theta), dtype=np.float32)

    best.update({
        "pinball_per_tau": pinball_per_tau.astype(np.float32),
        "pinball_per_dim": pinball_per_dim.astype(np.float32),
        "crossing_rate": float(crossing_rate),
        "calibration": cov.astype(np.float32),
        "iqr_val": iqr_val.astype(np.float32),
    })
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="two_moons_distractors")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/probe/cache")
    ap.add_argument("--out-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/quantile")
    ap.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taus = tuple(float(t) for t in args.taus)
    taus_arr = np.asarray(taus, dtype=np.float32)
    n_tau = len(taus)

    pinball = np.zeros(N_LAYERS)
    pinball_baseline = np.zeros(N_LAYERS)
    pinball_per_tau = np.zeros((N_LAYERS, n_tau))
    crossing_rate = np.zeros(N_LAYERS)
    calibration = np.zeros((N_LAYERS, n_tau))
    mean_alpha = np.zeros(N_LAYERS)
    quantile_alpha = np.zeros(N_LAYERS)

    pred_quantiles_val: np.ndarray | None = None
    pinball_per_dim: np.ndarray | None = None
    iqr_val: np.ndarray | None = None
    thetas_val_saved: np.ndarray | None = None

    shared_data: dict | None = None
    need_sim = any(
        not (cache / f"{args.task}_n{args.n_train}_layer{k}_s{args.seed}.npz").exists()
        for k in range(N_LAYERS)
    )
    if need_sim:
        print(f"Simulating {args.task} once for all layers...")
        shared_data = simulate(args.task, args.n_train, args.n_val, args.seed)

    print(f"Quantile probe sweep: task={args.task} seed={args.seed} taus={taus}")
    for k in range(N_LAYERS):
        e_tr, e_va, th_tr, th_va = extract_or_load(
            args.task, args.n_train, args.n_val, args.seed, k, cache,
            data=shared_data,
        )
        mean_best = fit_mean_probe(e_tr, th_tr, e_va, th_va)
        q_best = fit_quantile_probe(
            e_tr, th_tr, e_va, th_va,
            alpha_mu=mean_best["alpha"], taus=taus,
        )
        if pred_quantiles_val is None:
            n_val_actual, dim_theta = th_va.shape
            pred_quantiles_val = np.zeros(
                (N_LAYERS, n_tau, n_val_actual, dim_theta), dtype=np.float32,
            )
            pinball_per_dim = np.zeros((N_LAYERS, dim_theta))
            iqr_val = np.zeros((N_LAYERS, n_val_actual, dim_theta), dtype=np.float32)
            thetas_val_saved = th_va.copy()

        pinball[k] = q_best["pinball"]
        pinball_baseline[k] = q_best["pinball_baseline"]
        pinball_per_tau[k] = q_best["pinball_per_tau"]
        pinball_per_dim[k] = q_best["pinball_per_dim"]
        crossing_rate[k] = q_best["crossing_rate"]
        calibration[k] = q_best["calibration"]
        pred_quantiles_val[k] = q_best["pred_quantiles_val"].transpose(1, 0, 2)
        iqr_val[k] = q_best["iqr_val"]
        mean_alpha[k] = mean_best["alpha"]
        quantile_alpha[k] = q_best["alpha"]

        print(
            f"  layer {k:2d}  pinball={pinball[k]:+.4f} "
            f"(baseline={pinball_baseline[k]:+.4f}) "
            f"crossing={crossing_rate[k]:.4f}  "
            f"α_μ={mean_alpha[k]}  α_q={quantile_alpha[k]}"
        )

    best_layer = int(np.argmin(pinball))
    print(f"\nBest pinball at layer {best_layer}: {pinball[best_layer]:.4f}  "
          f"(baseline={pinball_baseline[best_layer]:.4f})")

    out_npz = out_dir / f"{args.task}_s{args.seed}.npz"
    out_png = out_dir / f"{args.task}_s{args.seed}.png"
    np.savez(
        str(out_npz),
        layers=np.arange(N_LAYERS), taus=taus_arr,
        pinball=pinball, pinball_baseline=pinball_baseline,
        pinball_per_tau=pinball_per_tau, pinball_per_dim=pinball_per_dim,
        crossing_rate=crossing_rate, calibration=calibration,
        pred_quantiles_val=pred_quantiles_val,
        iqr_val=iqr_val, thetas_val=thetas_val_saved,
        mean_alpha=mean_alpha, quantile_alpha=quantile_alpha,
        seed=args.seed, task=args.task, best_layer=best_layer,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    layers = np.arange(N_LAYERS)

    ax = axes[0, 0]
    ax.plot(layers, pinball, marker="o", lw=2, color="C3", label="quantile probe")
    ax.plot(layers, pinball_baseline, marker="s", lw=1.5, color="grey",
            ls="--", label="constant-quantile baseline")
    for t_i, tau in enumerate(taus):
        ax.plot(layers, pinball_per_tau[:, t_i], marker=".", color="C3",
                alpha=0.4, label=f"τ={tau}" if t_i < 3 else None)
    ax.axvline(best_layer, color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("Encoder layer"); ax.set_ylabel("Pinball loss")
    ax.set_title("A. Pinball vs layer")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if (0.25 in taus_arr) and (0.75 in taus_arr):
        med = np.median(iqr_val, axis=1).mean(axis=-1)
        lo = np.quantile(iqr_val, 0.25, axis=1).mean(axis=-1)
        hi = np.quantile(iqr_val, 0.75, axis=1).mean(axis=-1)
        ax.plot(layers, med, marker="o", lw=2, color="C0", label="median IQR (val)")
        ax.fill_between(layers, lo, hi, color="C0", alpha=0.2,
                        label="IQR of IQR")
        ax.axvline(best_layer, color="grey", ls=":", alpha=0.7)
        ax.set_xlabel("Encoder layer")
        ax.set_ylabel("Predicted q₀.₇₅ − q₀.₂₅ (val)")
        ax.set_title("B. Posterior-scale (IQR) emergence")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        ax.set_axis_off()

    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], color="grey", ls="--", alpha=0.6, label="ideal")
    ax.plot(taus_arr, calibration[best_layer], "o-", color="C2", lw=2,
            label=f"layer {best_layer}")
    ax.set_xlabel("Nominal quantile τ"); ax.set_ylabel("Empirical coverage")
    ax.set_title("C. Calibration at best layer")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1, 1]
    n_pick = min(10, thetas_val_saved.shape[0])
    order = np.argsort(thetas_val_saved[:, 0])
    pick = order[np.linspace(0, len(order) - 1, n_pick).astype(int)]
    pq_pick = pred_quantiles_val[best_layer][:, pick, :]
    xs = np.arange(n_pick)
    th_sorted = thetas_val_saved[pick, 0]
    n_band = n_tau // 2
    for b_i in range(n_band):
        lo_b = pq_pick[b_i, :, 0]
        hi_b = pq_pick[n_tau - 1 - b_i, :, 0]
        ax.fill_between(xs, lo_b, hi_b, color="C3",
                        alpha=0.15 + 0.20 * b_i / max(n_band - 1, 1),
                        label=f"τ ∈ [{taus[b_i]}, {taus[n_tau - 1 - b_i]}]")
    if n_tau % 2 == 1:
        ax.plot(xs, pq_pick[n_tau // 2, :, 0], "o-", color="C3",
                label=f"median (τ={taus[n_tau // 2]})")
    ax.plot(xs, th_sorted, "k.", ms=8, label="true θ₀")
    ax.set_xlabel("Val example (sorted by θ₀)")
    ax.set_ylabel("Predicted q_τ for θ₀")
    ax.set_title(f"D. Fan plot at best layer ({best_layer}), θ dim 0")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Quantile probe: {args.task} | seed={args.seed} "
                 f"| τ={list(taus)} | crossing={crossing_rate[best_layer]:.3f}")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=140)
    print(f"Wrote {out_png} and {out_npz}")


if __name__ == "__main__":
    main()
