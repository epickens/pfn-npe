"""Run mean + variance + quantile probes on raw vs contrastive embeddings.

For each (task, seed): extract embeddings with raw TabPFN AND with the
contrastive encoder; run fit_mean_probe, fit_variance_probe, and
fit_quantile_probe on each; report:

  - mean R² (linear θ-decoding)
  - σ² NLL (heteroscedastic Gaussian)
  - pinball (multi-τ quantile regression)

Comparable linear-probe R² and σ² NLL across encoders indicate that the v3
representation preserves θ-relevant information even when the downstream NSF
does not exploit it at the same training budget. Degraded probe metrics indicate
loss of θ-relevant structure in the encoder representation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402
from scripts.layer_linear_probe import (  # noqa: E402
    fit_mean_probe, fit_variance_probe,
)
from scripts.layer_quantile_probe import fit_quantile_probe  # noqa: E402


def get_embeddings(task: str, seed: int, n_train: int, n_val: int,
                   data: dict, encoder_ckpt: str | None
                   ) -> tuple[np.ndarray, np.ndarray]:
    emb = TabPFNEmbedder(
        context_size=1000, seed=seed, label_strategy="per_dim",
        layer=None, model_type="regressor",
    )
    emb.fit(data["xs_train"], thetas=data["thetas_train"])
    if encoder_ckpt:
        emb.load_encoder_checkpoint(encoder_ckpt)
    e_tr = emb.transform(data["xs_train"]).astype(np.float32, copy=False)
    e_va = emb.transform(data["xs_val"]).astype(np.float32, copy=False)
    return e_tr, e_va


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--encoder-checkpoint", required=True)
    ap.add_argument("--encoder-tag", default="v3")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"task={args.task} seed={args.seed}")
    print(f"Simulating ({args.n_train} train, {args.n_val} val)...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    th_tr, th_va = data["thetas_train"], data["thetas_val"]
    print(f"  dim_theta={data['dim_theta']}, dim_x={data['dim_x']}")

    print("\nExtracting raw TabPFN embeddings...")
    e_tr_raw, e_va_raw = get_embeddings(
        args.task, args.seed, args.n_train, args.n_val, data, None,
    )
    print(f"\nExtracting {args.encoder_tag} embeddings...")
    e_tr_v3, e_va_v3 = get_embeddings(
        args.task, args.seed, args.n_train, args.n_val, data, args.encoder_checkpoint,
    )

    results: dict[str, dict] = {}
    for label, (e_tr, e_va) in [("raw", (e_tr_raw, e_va_raw)),
                                 (args.encoder_tag, (e_tr_v3, e_va_v3))]:
        print(f"\n[{label}] mean probe...")
        mean = fit_mean_probe(e_tr, th_tr, e_va, th_va)
        print(f"  α={mean['alpha']}, R²={mean['r2']:.4f}, "
              f"per-dim R²={[f'{r:.3f}' for r in mean['r2_per_dim']]}")
        print(f"[{label}] variance probe...")
        var = fit_variance_probe(e_tr, th_tr, e_va, th_va,
                                 alpha_mu=mean["alpha"])
        print(f"  α_σ={var['alpha']}, NLL={var['nll']:.4f} "
              f"(homo={var['nll_homo']:.4f}, Δ={var['nll'] - var['nll_homo']:+.4f})")
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

    print("\n=== Summary: encoder linear-probe comparison ===")
    print(f"{'metric':<22} {'raw':>10} {args.encoder_tag:>14} {'Δ':>10}")
    for k, prefix in [("r2", "mean R²"),
                       ("nll", "Gaussian NLL"),
                       ("pinball", "pinball loss")]:
        a = results["raw"][k]; b = results[args.encoder_tag][k]
        d = b - a
        print(f"{prefix:<22} {a:>10.4f} {b:>14.4f} {d:>+10.4f}")
    print(f"{'NLL homo (ref)':<22} {results['raw']['nll_homo']:>10.4f} "
          f"{results[args.encoder_tag]['nll_homo']:>14.4f}")
    print(f"{'pinball baseline':<22} {results['raw']['pinball_baseline']:>10.4f} "
          f"{results[args.encoder_tag]['pinball_baseline']:>14.4f}")


if __name__ == "__main__":
    main()
