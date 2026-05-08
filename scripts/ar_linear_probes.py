"""Compare raw TabPFN encoder vs AR hidden-state embeddings on linear probes.

For each (task, seed):
  1. Extract raw TabPFN encoder per-dim concatenated embeddings on
     (x_train, x_val) — same as `scripts/encoder_linear_probes.py`.
  2. Fit `TabPFNAR` on (theta_train, x_train) and extract concatenated
     AR hidden states for (theta_val, x_val) at the last encoder layer
     (default: 11). Each val row gets its own (x_i, theta_i_<d) prefix.
  3. Run `fit_mean_probe`, `fit_variance_probe`, `fit_quantile_probe` on
     each. Report side-by-side.

Scientific note: AR per-dim hidden state h_d(x, theta_<d) carries
information about theta_<d directly (it's an input). For predicting
theta_d, the AR probe has *additional* prefix info the encoder lacks —
so a higher mean R² is expected on the chained dims and partly reflects
chain-rule advantage rather than pure encoder quality. The variance
probe NLL is the more honest signal: how tightly conditioned on
(x, theta_<d) the predicted distribution of theta_d is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.ar_density import TabPFNAR  # noqa: E402
from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402
from scripts.layer_linear_probe import (  # noqa: E402
    fit_mean_probe, fit_variance_probe,
)
from scripts.layer_quantile_probe import fit_quantile_probe  # noqa: E402


def get_encoder_embeddings(seed: int, data: dict, context_size: int = 1000,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Raw TabPFN per-dim concatenated encoder, last layer.

    `context_size` defaults to 1000 (TabPFNEmbedder's standard); pass
    10000 to match TabPFNAR's full-context fit and remove the
    context-size confound from AR-vs-encoder comparisons.
    """
    emb = TabPFNEmbedder(
        context_size=context_size, seed=seed, label_strategy="per_dim",
        layer=None, model_type="regressor",
    )
    emb.fit(data["xs_train"], thetas=data["thetas_train"])
    e_tr = emb.transform(data["xs_train"]).astype(np.float32, copy=False)
    e_va = emb.transform(data["xs_val"]).astype(np.float32, copy=False)
    return e_tr, e_va


def get_ar_hidden_states(seed: int, data: dict, layer: int = 11,
                         prefix_mode: str = "self",
                         ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-dim AR hidden states.

    Returns lists of per-dim arrays (one entry per θ-dim). Caller can
    concatenate for the joint probe or use individual entries for
    per-dim probes.

    prefix_mode: 'self' | 'mean' | 'zero' — see TabPFNAR.hidden_states.
    """
    task = get_task(data["task"]) if "task" in data else None
    prior_dist = task.get_prior_dist() if task is not None else None

    ar = TabPFNAR(prior=prior_dist, seed=seed)
    ar.fit(data["thetas_train"], data["xs_train"])

    print(f"  extracting AR hidden states for {data['xs_train'].shape[0]} train + "
          f"{data['xs_val'].shape[0]} val rows at layer {layer} "
          f"(prefix_mode={prefix_mode!r}) ...")
    per_dim_tr = ar.hidden_states(
        data["thetas_train"], data["xs_train"], layer=layer,
        prefix_mode=prefix_mode, return_per_dim=True,
    )
    per_dim_va = ar.hidden_states(
        data["thetas_val"], data["xs_val"], layer=layer,
        prefix_mode=prefix_mode, return_per_dim=True,
    )
    per_dim_tr = [p.astype(np.float32, copy=False) for p in per_dim_tr]
    per_dim_va = [p.astype(np.float32, copy=False) for p in per_dim_va]
    return per_dim_tr, per_dim_va


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--ar-layer", type=int, default=11,
                    help="Encoder layer to extract AR hidden states at.")
    ap.add_argument("--prefix-mode", default="self",
                    choices=["self", "mean", "zero"],
                    help="AR prefix override at extraction. 'self' is "
                         "leakage-prone; 'mean' isolates encoder.")
    ap.add_argument("--encoder-context-size", type=int, default=1000,
                    help="Context size for TabPFNEmbedder. Default 1000; "
                         "set 10000 to match AR's full-context fit.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"task={args.task} seed={args.seed}")
    print(f"Simulating ({args.n_train} train, {args.n_val} val)...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    data["task"] = args.task   # for downstream prior lookup
    th_tr, th_va = data["thetas_train"], data["thetas_val"]
    print(f"  dim_theta={data['dim_theta']}, dim_x={data['dim_x']}")

    print(f"\nExtracting raw TabPFN encoder embeddings "
          f"(context_size={args.encoder_context_size})...")
    e_tr_enc, e_va_enc = get_encoder_embeddings(
        args.seed, data, context_size=args.encoder_context_size,
    )
    emb_per_dim = e_tr_enc.shape[1] // data["dim_theta"]
    enc_per_dim_tr = [
        e_tr_enc[:, d * emb_per_dim:(d + 1) * emb_per_dim]
        for d in range(data["dim_theta"])
    ]
    enc_per_dim_va = [
        e_va_enc[:, d * emb_per_dim:(d + 1) * emb_per_dim]
        for d in range(data["dim_theta"])
    ]
    print(f"  encoder emb shape: train {e_tr_enc.shape}, val {e_va_enc.shape}; "
          f"per-dim slice = {emb_per_dim}")

    print(f"\nExtracting AR hidden states "
          f"(layer={args.ar_layer}, prefix_mode={args.prefix_mode!r})...")
    ar_per_dim_tr, ar_per_dim_va = get_ar_hidden_states(
        args.seed, data, layer=args.ar_layer, prefix_mode=args.prefix_mode,
    )
    e_tr_ar = np.concatenate(ar_per_dim_tr, axis=1)
    e_va_ar = np.concatenate(ar_per_dim_va, axis=1)
    print(f"  AR hidden shape: train {e_tr_ar.shape}, val {e_va_ar.shape}")

    results: dict[str, dict] = {}
    for label, (e_tr, e_va) in [("encoder", (e_tr_enc, e_va_enc)),
                                 ("ar", (e_tr_ar, e_va_ar))]:
        print(f"\n[{label}] mean probe...")
        mean = fit_mean_probe(e_tr, th_tr, e_va, th_va)
        print(f"  α={mean['alpha']}, R²={mean['r2']:.4f}, "
              f"per-dim R²={[f'{r:.3f}' for r in mean['r2_per_dim']]}")
        print(f"[{label}] variance probe...")
        var = fit_variance_probe(e_tr, th_tr, e_va, th_va,
                                 alpha_mu=mean["alpha"])
        print(f"  α_σ={var['alpha']}, NLL={var['nll']:.4f} "
              f"(homo={var['nll_homo']:.4f}, "
              f"Δ={var['nll'] - var['nll_homo']:+.4f})")
        print(f"[{label}] quantile probe (τ=5 levels)...")
        q = fit_quantile_probe(e_tr, th_tr, e_va, th_va,
                               alpha_mu=mean["alpha"])
        print(f"  α_q={q['alpha']}, pinball={q['pinball']:.4f} "
              f"(baseline={q['pinball_baseline']:.4f})")
        results[label] = {
            "r2": mean["r2"],
            "r2_per_dim": mean["r2_per_dim"],
            "nll": var["nll"],
            "nll_homo": var["nll_homo"],
            "pinball": q["pinball"],
            "pinball_baseline": q["pinball_baseline"],
        }

    print("\n=== Summary: encoder vs AR hidden states (concat probe) ===")
    print(f"{'metric':<22} {'encoder':>10} {'AR':>10} {'Δ (AR - enc)':>14}")
    for k, prefix in [("r2", "mean R²"),
                      ("nll", "Gaussian NLL"),
                      ("pinball", "pinball loss")]:
        a = results["encoder"][k]
        b = results["ar"][k]
        d = b - a
        print(f"{prefix:<22} {a:>10.4f} {b:>10.4f} {d:>+14.4f}")
    print(f"{'NLL homo (ref)':<22} {results['encoder']['nll_homo']:>10.4f} "
          f"{results['ar']['nll_homo']:>10.4f}")
    print(f"{'pinball baseline':<22} {results['encoder']['pinball_baseline']:>10.4f} "
          f"{results['ar']['pinball_baseline']:>10.4f}")

    # ── Per-dim probes ────────────────────────────────────────────────────
    # Standalone single-θ-dim probe (mean R² + homoscedastic NLL via 5-fold
    # OOF residuals). Avoids the multi-output assumptions in
    # `fit_variance_probe`. Targets θ_d as a 1D vector.
    from sklearn.linear_model import Ridge as _Ridge
    from sklearn.metrics import r2_score as _r2
    from sklearn.model_selection import KFold as _KFold

    def per_dim_probe(e_tr_d, e_va_d, t_tr_1d, t_va_1d,
                      alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
        best = {"r2": -np.inf, "alpha": None}
        for a in alphas:
            m = _Ridge(alpha=a).fit(e_tr_d, t_tr_1d)
            pred = m.predict(e_va_d)
            r2 = float(_r2(t_va_1d, pred))
            if r2 > best["r2"]:
                best.update({"r2": r2, "alpha": a, "mu_va": pred})
        kf = _KFold(n_splits=5, shuffle=True, random_state=0)
        mu_oof = np.zeros_like(t_tr_1d)
        for tr_idx, ho_idx in kf.split(e_tr_d):
            m = _Ridge(alpha=best["alpha"]).fit(e_tr_d[tr_idx], t_tr_1d[tr_idx])
            mu_oof[ho_idx] = m.predict(e_tr_d[ho_idx])
        sigma_sq_homo = float(((t_tr_1d - mu_oof) ** 2).mean()) + 1e-6
        r_va_sq = (t_va_1d - best["mu_va"]) ** 2
        log_2pi = float(np.log(2 * np.pi))
        nll_homo = float(
            0.5 * (r_va_sq / sigma_sq_homo + np.log(sigma_sq_homo) + log_2pi).mean()
        )
        return {"r2": best["r2"], "nll_homo": nll_homo,
                "sigma_sq_homo": sigma_sq_homo}

    print("\n=== Per-dim mean probes (target = θ_d, 1D) ===")
    print(f"{'dim':>3} {'enc R²':>10} {'AR R²':>10} {'Δ R²':>9} "
          f"{'enc NLL_homo':>14} {'AR NLL_homo':>13} {'Δ NLL_homo':>12}")
    for d in range(data["dim_theta"]):
        t_tr = th_tr[:, d].astype(np.float32)
        t_va = th_va[:, d].astype(np.float32)
        e = per_dim_probe(enc_per_dim_tr[d], enc_per_dim_va[d], t_tr, t_va)
        a = per_dim_probe(ar_per_dim_tr[d], ar_per_dim_va[d], t_tr, t_va)
        print(f"{d:>3} {e['r2']:>+10.4f} {a['r2']:>+10.4f} "
              f"{a['r2'] - e['r2']:>+9.4f} "
              f"{e['nll_homo']:>+14.4f} {a['nll_homo']:>+13.4f} "
              f"{a['nll_homo'] - e['nll_homo']:>+12.4f}")


if __name__ == "__main__":
    main()
