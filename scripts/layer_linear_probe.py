"""Linear probes of TabPFN embeddings per layer.

For each of the 12 encoder layers:
 1. Build a TabPFNEmbedder matching the benchmark config (per_dim, regressor).
 2. Extract embeddings on train+val splits. Cache to disk so reruns are cheap.
 3. Fit a mean probe: ridge regression θ | embedding, α swept.
 4. (Optional) Fit a heteroscedastic variance probe: two linear heads (μ and
    log σ²) trained jointly to minimize Gaussian NLL. Reports per-layer NLL
    alongside a homoscedastic baseline.

Writes a per-layer npz + a 2-panel summary png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder, simulate  # noqa: E402

N_LAYERS = 12
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
EPS = 1e-6
LOG_2PI = float(np.log(2 * np.pi))


class LinearHead:
    """Minimal linear model exposing a scikit-style ``.predict`` interface.

    Used to pass the jointly-trained μ and log σ² heads out of the variance
    probe in the same shape as the sklearn Ridge models used for the mean probe.
    """

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = W
        self.b = b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b


def extract_or_load(
    task: str, n_train: int, n_val: int, seed: int, layer: int, cache: Path,
    data: dict | None = None,
    model_version: str = "v2",
    label_strategy: str = "per_dim",
):
    cache.mkdir(parents=True, exist_ok=True)
    # Cache-key suffix only when non-default, so existing v2/per_dim caches
    # (240 files) still hit on the legacy filename.
    suffix_parts = []
    if model_version != "v2":
        suffix_parts.append(f"_m{model_version.replace('.', '')}")
    if label_strategy != "per_dim":
        suffix_parts.append(f"_ls{label_strategy}")
    suffix = "".join(suffix_parts)
    key = cache / f"{task}_n{n_train}_layer{layer}_s{seed}{suffix}.npz"
    if key.exists():
        d = np.load(key)
        return d["emb_train"], d["emb_val"], d["thetas_train"], d["thetas_val"]

    print(f"  [layer {layer}] extracting (cache miss, mv={model_version}, ls={label_strategy})")
    if data is None:
        data = simulate(task, n_train, n_val, seed)
    emb = TabPFNEmbedder(
        context_size=1000, seed=seed, label_strategy=label_strategy,
        layer=layer, model_type="regressor", model_version=model_version,
    )
    emb.fit(data["xs_train"], thetas=data["thetas_train"])
    e_tr = emb.transform(data["xs_train"])
    e_va = emb.transform(data["xs_val"])
    np.savez(
        str(key),
        emb_train=e_tr, emb_val=e_va,
        thetas_train=data["thetas_train"], thetas_val=data["thetas_val"],
    )
    return e_tr, e_va, data["thetas_train"], data["thetas_val"]


def fit_mean_probe(e_tr: np.ndarray, th_tr: np.ndarray,
                   e_va: np.ndarray, th_va: np.ndarray) -> dict:
    best: dict = {"alpha": None, "r2": -np.inf, "r2_per_dim": None,
                  "mu_val": None, "model": None}
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(e_tr, th_tr)
        pred = m.predict(e_va)
        r2 = float(r2_score(th_va, pred, multioutput="uniform_average"))
        if r2 > best["r2"]:
            best = {
                "alpha": a, "r2": r2,
                "r2_per_dim": r2_score(th_va, pred, multioutput="raw_values").tolist(),
                "mu_val": pred,
                "model": m,
            }
    return best


def fit_variance_probe(e_tr: np.ndarray, th_tr: np.ndarray,
                       e_va: np.ndarray, th_va: np.ndarray,
                       alpha_mu: float = 1.0,
                       alphas_sigma: tuple = ALPHAS,
                       n_epochs: int = 500,
                       lr: float = 5e-3) -> dict:
    """Heteroscedastic variance probe on top of a fixed ridge μ.

    Fits a linear log-variance head by minimizing Gaussian NLL. The μ head
    is held fixed at the ridge fit (α = ``alpha_mu``, pass the best α from
    the mean probe) — the mean probe's ridge is already the best linear μ
    we can get, so the variance probe only needs to fit σ² on top.

    The log-variance head is initialized at zero weights + bias = log σ²_homo
    so initial val NLL equals the homoscedastic baseline. The α sweep over
    ``alphas_sigma`` includes an implicit "α = ∞" (pure homoscedastic) state
    by initializing ``best`` to the homo fit. This guarantees the returned
    NLL is ≤ homo NLL by construction: if no heteroscedastic config beats it,
    we report the constant-σ² solution.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_tr_t = torch.tensor(e_tr, dtype=torch.float32, device=device)
    th_tr_t = torch.tensor(th_tr, dtype=torch.float32, device=device)
    e_va_t = torch.tensor(e_va, dtype=torch.float32, device=device)
    th_va_t = torch.tensor(th_va, dtype=torch.float32, device=device)

    D = e_tr.shape[1]
    dim_theta = th_tr.shape[1]
    n_tr = e_tr.shape[0]

    # μ head: fixed to ridge warm-start.
    mu_ridge = Ridge(alpha=alpha_mu).fit(e_tr, th_tr)
    W_mu_np = mu_ridge.coef_.T.astype(np.float32)        # (D, dim_theta)
    b_mu_np = mu_ridge.intercept_.astype(np.float32)     # (dim_theta,)
    W_mu_t = torch.tensor(W_mu_np, device=device)
    b_mu_t = torch.tensor(b_mu_np, device=device)
    mu_val_np = e_va @ W_mu_np + b_mu_np                 # (n_val, dim_theta)

    # Homoscedastic baseline: 5-fold OOF residuals → per-dim constant σ².
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    mu_oof = np.zeros_like(th_tr)
    for tr_idx, ho_idx in kf.split(e_tr):
        m = Ridge(alpha=alpha_mu).fit(e_tr[tr_idx], th_tr[tr_idx])
        mu_oof[ho_idx] = m.predict(e_tr[ho_idx])
    sigma_sq_homo = ((th_tr - mu_oof) ** 2).mean(axis=0) + EPS  # (dim_theta,)
    log_sigma_sq_homo_np = np.log(sigma_sq_homo).astype(np.float32)
    log_sigma_sq_homo_t = torch.tensor(log_sigma_sq_homo_np, device=device)

    r_val = (th_va - mu_val_np) ** 2
    nll_homo_ed = 0.5 * (r_val / sigma_sq_homo + np.log(sigma_sq_homo) + LOG_2PI)
    nll_homo = float(nll_homo_ed.mean())
    log_var_val_homo = np.broadcast_to(
        log_sigma_sq_homo_np[None, :], (th_va.shape[0], dim_theta),
    ).copy()

    mu_head = LinearHead(W_mu_np, b_mu_np)
    log_var_head_homo = LinearHead(
        np.zeros((D, dim_theta), dtype=np.float32),
        log_sigma_sq_homo_np,
    )

    # Initialize best to the pure-homoscedastic solution. Any α_σ that
    # doesn't beat it gets discarded; this makes the probe fail-safe.
    best: dict = {
        "alpha": np.inf, "nll": nll_homo,
        "nll_per_dim": nll_homo_ed.mean(axis=0).tolist(),
        "log_var_val": log_var_val_homo,
        "mu_val": mu_val_np,
        "nll_homo": nll_homo,
        "mu_model": mu_head,
        "log_var_model": log_var_head_homo,
        "sigma_sq_homo": sigma_sq_homo,
    }

    for a_sigma in alphas_sigma:
        W_lv = torch.nn.Parameter(torch.zeros(D, dim_theta, device=device))
        b_lv = torch.nn.Parameter(log_sigma_sq_homo_t.clone())
        opt = torch.optim.Adam([W_lv, b_lv], lr=lr)

        with torch.no_grad():
            mu_tr = e_tr_t @ W_mu_t + b_mu_t
            resid_sq_tr = (th_tr_t - mu_tr) ** 2

        for _ in range(n_epochs):
            lv = e_tr_t @ W_lv + b_lv
            nll = 0.5 * (resid_sq_tr * torch.exp(-lv) + lv + LOG_2PI)
            l2 = (a_sigma / n_tr) * (W_lv ** 2).sum()
            loss = nll.mean() + l2
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            mu_va_t = e_va_t @ W_mu_t + b_mu_t
            lv_va = e_va_t @ W_lv + b_lv
            r_va = (th_va_t - mu_va_t) ** 2
            nll_ed = 0.5 * (r_va * torch.exp(-lv_va) + lv_va + LOG_2PI)
            val_nll = float(nll_ed.mean())

            if val_nll < best["nll"]:
                best = {
                    "alpha": a_sigma, "nll": val_nll,
                    "nll_per_dim": nll_ed.mean(dim=0).cpu().numpy().tolist(),
                    "log_var_val": lv_va.cpu().numpy(),
                    "mu_val": mu_val_np,
                    "nll_homo": nll_homo,
                    "mu_model": mu_head,
                    "log_var_model": LinearHead(
                        W_lv.detach().cpu().numpy(),
                        b_lv.detach().cpu().numpy(),
                    ),
                    "sigma_sq_homo": sigma_sq_homo,
                }
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="two_moons_distractors")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default="pfn_testing/sbi/outputs/layer_ablation/probe/cache")
    ap.add_argument("--out-prefix", default="pfn_testing/sbi/outputs/layer_ablation/probe/")
    ap.add_argument("--probe-variance", action="store_true", default=True,
                    help="Also fit heteroscedastic variance probe (default: true)")
    ap.add_argument("--no-probe-variance", dest="probe_variance", action="store_false")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if output npz exists.")
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"])
    args = ap.parse_args()

    # Model-version-aware output paths so v2 / v2.5 don't collide.
    mv_suffix = (
        f"_m{args.model_version.replace('.', '')}"
        if args.model_version != "v2" else ""
    )
    out_npz_check = Path(f"{args.out_prefix}{args.task}_s{args.seed}{mv_suffix}.npz")
    if out_npz_check.exists() and not args.force:
        print(f"[skip] {out_npz_check}")
        return

    cache = Path(args.cache_dir)
    r2 = np.zeros(N_LAYERS)
    per_dim: np.ndarray | None = None
    alpha_chosen = np.zeros(N_LAYERS)

    nll = np.zeros(N_LAYERS) if args.probe_variance else None
    nll_homo = np.zeros(N_LAYERS) if args.probe_variance else None
    log_var_alpha = np.zeros(N_LAYERS) if args.probe_variance else None
    nll_per_dim: np.ndarray | None = None
    mu_val_all: np.ndarray | None = None
    log_var_val_all: np.ndarray | None = None
    thetas_val_saved: np.ndarray | None = None

    shared_data: dict | None = None
    need_sim = any(
        not (cache / f"{args.task}_n{args.n_train}_layer{k}_s{args.seed}.npz").exists()
        for k in range(N_LAYERS)
    )
    if need_sim:
        print(f"Simulating {args.task} once for all layers...")
        shared_data = simulate(args.task, args.n_train, args.n_val, args.seed)
        print(f"  dim_theta={shared_data['dim_theta']}, dim_x={shared_data['dim_x']}")

    print(f"Linear probe sweep: task={args.task} seed={args.seed} "
          f"variance={args.probe_variance}")
    for k in range(N_LAYERS):
        e_tr, e_va, th_tr, th_va = extract_or_load(
            args.task, args.n_train, args.n_val, args.seed, k, cache,
            data=shared_data,
            model_version=args.model_version,
        )
        mean_best = fit_mean_probe(e_tr, th_tr, e_va, th_va)
        if per_dim is None:
            per_dim = np.zeros((N_LAYERS, len(mean_best["r2_per_dim"])))
        r2[k] = mean_best["r2"]
        per_dim[k] = mean_best["r2_per_dim"]
        alpha_chosen[k] = mean_best["alpha"]

        if args.probe_variance:
            var_best = fit_variance_probe(
                e_tr, th_tr, e_va, th_va, alpha_mu=mean_best["alpha"],
            )
            if mu_val_all is None:
                n_val_actual, dim_theta = th_va.shape
                mu_val_all = np.zeros((N_LAYERS, n_val_actual, dim_theta))
                log_var_val_all = np.zeros_like(mu_val_all)
                nll_per_dim = np.zeros((N_LAYERS, dim_theta))
                thetas_val_saved = th_va.copy()
            nll[k] = var_best["nll"]
            nll_homo[k] = var_best["nll_homo"]
            log_var_alpha[k] = var_best["alpha"]
            nll_per_dim[k] = var_best["nll_per_dim"]
            mu_val_all[k] = var_best["mu_val"]
            log_var_val_all[k] = var_best["log_var_val"]
            print(f"  layer {k:2d}  R²={r2[k]:+.4f}  "
                  f"NLL={nll[k]:+.4f} (homo={nll_homo[k]:+.4f})  "
                  f"α_μ={alpha_chosen[k]}  α_σ={log_var_alpha[k]}")
        else:
            print(f"  layer {k:2d}  R²={r2[k]:+.4f}  "
                  f"per-dim={per_dim[k]}  α={alpha_chosen[k]}")

    assert per_dim is not None
    best_layer_mean = int(np.argmax(r2))
    print(f"\nBest mean R² at layer {best_layer_mean}: {r2[best_layer_mean]:.4f}")
    if args.probe_variance:
        best_layer_var = int(np.argmin(nll))
        print(f"Best variance NLL at layer {best_layer_var}: "
              f"{nll[best_layer_var]:.4f}  (homo at same layer: {nll_homo[best_layer_var]:.4f})")

    out_npz = Path(f"{args.out_prefix}{args.task}_s{args.seed}{mv_suffix}.npz")
    out_png = Path(f"{args.out_prefix}{args.task}_s{args.seed}{mv_suffix}.png")
    save_kwargs: dict = dict(
        layers=np.arange(N_LAYERS),
        r2=r2, per_dim=per_dim, alphas=alpha_chosen,
        seed=args.seed, task=args.task,
    )
    if args.probe_variance:
        save_kwargs.update(
            nll=nll, nll_homo=nll_homo,
            nll_per_dim=nll_per_dim, log_var_alpha=log_var_alpha,
            mu_val=mu_val_all, log_var_val=log_var_val_all,
            thetas_val=thetas_val_saved,
        )
    np.savez(str(out_npz), **save_kwargs)

    # ── Plot ───────────────────────────────────────────────────────────────
    if args.probe_variance:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(range(N_LAYERS), r2, marker="o", lw=2, label="mean R² across θ dims")
    for d in range(per_dim.shape[1]):
        ax1.plot(range(N_LAYERS), per_dim[:, d], marker=".", alpha=0.5,
                 label=f"θ dim {d}")
    ax1.axhline(0, color="k", ls=":")
    ax1.set_xlabel("TabPFN encoder layer")
    ax1.set_ylabel("Ridge val R² (θ | embedding)")
    ax1.set_title("A. Mean probe")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    if args.probe_variance:
        assert nll is not None and nll_homo is not None and nll_per_dim is not None
        ax2.plot(range(N_LAYERS), nll, marker="o", lw=2, color="C3",
                 label="heteroscedastic probe (mean across θ)")
        ax2.plot(range(N_LAYERS), nll_homo, marker="s", lw=1.5, color="grey",
                 ls="--", label="homoscedastic baseline")
        for d in range(nll_per_dim.shape[1]):
            ax2.plot(range(N_LAYERS), nll_per_dim[:, d], marker=".",
                     color="C3", alpha=0.35,
                     label=f"θ dim {d}" if d < 3 else None)
        ax2.set_xlabel("TabPFN encoder layer")
        ax2.set_ylabel("Gaussian NLL (lower = better)")
        ax2.set_title("B. Variance probe")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Linear probes: {args.task} | seed={args.seed} "
                 f"| per_dim + regressor, target pool")
    fig.tight_layout(); fig.savefig(str(out_png), dpi=120)
    print(f"Wrote {out_png} and {out_npz}")


if __name__ == "__main__":
    main()
