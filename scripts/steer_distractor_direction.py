"""Causal localization of distractor filtering via CAA-style steering.

For each encoder layer k, compute a per-layer distractor direction:
  d[k] = mean(A_dist[k]) − mean(A_clean[k])
where A_*[k] are test-row activations from matched distract/clean inputs
(same θ, distractors zeroed in the clean version). d[k] has the same shape
as a single test row's representation at layer k.

Causal ablation: during a forward pass on distract inputs, project out d[k]
from each test row's activation at layer k. Continue forward. Measure the
downstream mean probe R² and variance probe NLL.

At layers where d[k] captures load-bearing distract content, projection
should move the probe output. At layers after filtering (d[k] residual), the
effect should saturate near zero.

This differs from whole-state patching in that it changes only one direction's
worth of content, leaving the rest of the representation intact — so the
forward pass can't simply re-derive the patched state from what remains.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402
from scripts.layer_linear_probe import (  # noqa: E402
    N_LAYERS, fit_mean_probe, fit_variance_probe,
)
from scripts.patch_distractor_filter import (  # noqa: E402
    _pool_target, _register_extraction_hooks, apply_probes,
    distractor_position_mask,
)


def compute_distractor_directions(
    embedder: TabPFNEmbedder,
    xs_dist: np.ndarray,
    xs_clean: np.ndarray,
) -> list[dict[int, torch.Tensor]]:
    """Return per-θ-dim per-layer distractor directions.

    For each θ-dim d and layer k:
      direction[d][k] has shape (n_features+1, emb_dim) =
        mean over calibration test rows of (A_dist[k] - A_clean[k])
    stored as float32 on the embedder's device.
    """
    assert embedder._per_dim_labels is not None
    per_dim_labels = embedder._per_dim_labels
    x_context = embedder._x_context
    n_ctx = embedder._n_context

    directions: list[dict[int, torch.Tensor]] = []

    for d, labels in enumerate(per_dim_labels):
        embedder.clf.fit(x_context, labels)
        embedder._apply_encoder_checkpoint()
        layers = embedder._get_transformer_model().transformer_encoder.layers

        dist_cap: dict = {}
        h_dist = _register_extraction_hooks(layers, list(range(N_LAYERS)), dist_cap)
        try:
            embedder.clf.get_embeddings(xs_dist, data_source="test")
        finally:
            for h in h_dist:
                h.remove()

        clean_cap: dict = {}
        h_clean = _register_extraction_hooks(layers, list(range(N_LAYERS)), clean_cap)
        try:
            embedder.clf.get_embeddings(xs_clean, data_source="test")
        finally:
            for h in h_clean:
                h.remove()

        d_layer = {}
        for k in range(N_LAYERS):
            a_dist = dist_cap[k][:, n_ctx:].float()      # (1, n_calib, n_feat+1, emb_dim)
            a_clean = clean_cap[k][:, n_ctx:].float()
            d_layer[k] = (a_dist - a_clean).mean(dim=(0, 1))  # (n_feat+1, emb_dim)
        directions.append(d_layer)

        # Release GPU memory held by captured activations
        dist_cap.clear()
        clean_cap.clear()

    return directions


def ablated_transform(
    embedder: TabPFNEmbedder,
    xs: np.ndarray,
    directions: list[dict[int, torch.Tensor]],
    ablate_layer: int,
    final_layer: int = 11,
) -> np.ndarray:
    """Forward pass on ``xs`` (distract) with the distractor direction at
    ``ablate_layer`` projected out of each test row. Returns (n_test,
    dim_theta * emb_dim) concatenated embedding.
    """
    assert embedder._per_dim_labels is not None
    per_dim_labels = embedder._per_dim_labels
    x_context = embedder._x_context
    n_ctx = embedder._n_context

    parts = []
    for d, labels in enumerate(per_dim_labels):
        embedder.clf.fit(x_context, labels)
        embedder._apply_encoder_checkpoint()
        layers = embedder._get_transformer_model().transformer_encoder.layers

        direction = directions[d][ablate_layer]          # (n_feat+1, emb_dim)
        d_flat_fp32 = direction.reshape(-1)              # (n_feat+1 * emb_dim,)
        d_norm_sq_fp32 = (d_flat_fp32 ** 2).sum() + 1e-8

        fin_cap: dict = {}

        def ablate_hook(module, input, output,
                        d_=d_flat_fp32, dn_=d_norm_sq_fp32, nc=n_ctx):
            # output: (1, n_ctx + n_test, n_feat+1, emb_dim), fp16 on GPU
            new_out = output.clone()
            test = new_out[:, nc:]                       # (1, n_test, n_feat+1, emb_dim)
            n_test = test.shape[1]
            test_flat = test.reshape(n_test, -1).float()  # cast for numerical stability
            coeffs = (test_flat @ d_) / dn_              # (n_test,)
            projection = coeffs.unsqueeze(1) * d_.unsqueeze(0)  # (n_test, dim_flat)
            test_ablated = (test_flat - projection).to(test.dtype)
            new_out[:, nc:] = test_ablated.reshape(test.shape)
            return new_out

        def capture_hook(module, input, output):
            fin_cap["out"] = output.detach().clone()

        h1 = layers[ablate_layer].register_forward_hook(ablate_hook)
        h2 = layers[final_layer].register_forward_hook(capture_hook)
        try:
            embedder.clf.get_embeddings(xs, data_source="test")
        finally:
            h1.remove()
            h2.remove()

        parts.append(_pool_target(fin_cap["out"], n_ctx))

    return np.concatenate(parts, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="two_moons_distractors")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-calib", type=int, default=500,
                    help="Calibration rows for computing d[k]")
    ap.add_argument("--cache-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/probe/cache")
    ap.add_argument("--out-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/steer")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    print(f"Simulating {args.task} (n_train={args.n_train}, n_val={args.n_val}, s={args.seed})")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    xs_train = data["xs_train"]
    xs_val = data["xs_val"]
    th_tr = data["thetas_train"]
    th_va = data["thetas_val"]
    print(f"  dim_theta={data['dim_theta']}  dim_x={data['dim_x']}")

    noise_mask = distractor_position_mask(args.task)
    print(f"  distractor positions: {int(noise_mask.sum())}/{noise_mask.size}")

    # Split calibration from evaluation within val so direction computation
    # doesn't look at the same rows we measure probe on.
    n_calib = min(args.n_calib, args.n_val // 2)
    xs_calib_dist = xs_val[:n_calib].copy()
    xs_calib_clean = xs_val[:n_calib].copy()
    xs_calib_clean[:, noise_mask] = 0.0

    xs_eval = xs_val[n_calib:]
    th_eval = th_va[n_calib:]
    print(f"  calibration: first {n_calib} val rows; evaluation: remaining {len(xs_eval)}")

    # Train probes on cached layer-11 distract embeddings (use full train split).
    key = cache / f"{args.task}_n{args.n_train}_layer11_s{args.seed}.npz"
    if not key.exists():
        print(f"Cache miss at {key}; run scripts/layer_linear_probe.py first.")
        sys.exit(1)
    d_cache = np.load(key)
    e_tr = d_cache["emb_train"]
    e_va = d_cache["emb_val"]
    print(f"Loaded cached embeddings: e_tr {e_tr.shape}, e_va {e_va.shape}")

    print("Fitting probes on distract train embeddings...")
    mean_best = fit_mean_probe(e_tr, th_tr, e_va, th_va)
    var_best = fit_variance_probe(e_tr, th_tr, e_va, th_va,
                                  alpha_mu=mean_best["alpha"])
    mean_model = mean_best["model"]
    lv_model = var_best["log_var_model"]
    mu_probe_model = var_best["mu_model"]
    # Baselines on the eval slice
    eval_cache_rows = slice(n_calib, None)
    e_eval = e_va[eval_cache_rows]
    baseline_r2, baseline_nll = apply_probes(
        e_eval, th_eval, mu_probe_model, lv_model,
    )
    print(f"  distract baseline on eval slice: R²={baseline_r2:+.4f}  NLL={baseline_nll:+.4f}")

    # Embedder (default path) for calibration + ablation
    print("Fitting embedder for steering...")
    embedder = TabPFNEmbedder(
        context_size=1000, seed=args.seed,
        label_strategy="per_dim", layer=None, model_type="regressor",
    )
    embedder.fit(xs_train, thetas=th_tr)

    print(f"Computing distractor directions ({n_calib} calib rows)...")
    directions = compute_distractor_directions(
        embedder, xs_calib_dist, xs_calib_clean,
    )
    # Direction magnitudes per layer: aggregate across θ-dims
    print("Direction norms per layer (fp32, flattened):")
    for k in range(N_LAYERS):
        norms = [directions[d][k].reshape(-1).norm().item()
                 for d in range(len(directions))]
        print(f"  layer {k:2d}  ||d||={np.mean(norms):.3f}  (per θ-dim: {[f'{n:.2f}' for n in norms]})")

    print("Running ablation sweep across 12 layers...")
    ablated_r2 = np.zeros(N_LAYERS)
    ablated_nll = np.zeros(N_LAYERS)
    for k in range(N_LAYERS):
        emb_k = ablated_transform(embedder, xs_eval, directions, k)
        r2, nll = apply_probes(emb_k, th_eval, mu_probe_model, lv_model)
        ablated_r2[k] = r2
        ablated_nll[k] = nll
        d_r2 = r2 - baseline_r2
        d_nll = nll - baseline_nll
        print(f"  layer {k:2d}  R²={r2:+.4f} (ΔR²={d_r2:+.4f})  "
              f"NLL={nll:+.4f} (ΔNLL={d_nll:+.4f})")

    delta_r2 = ablated_r2 - baseline_r2
    delta_nll = ablated_nll - baseline_nll

    out_npz = out_dir / f"{args.task}_s{args.seed}.npz"
    np.savez(str(out_npz),
             layers=np.arange(N_LAYERS),
             ablated_r2=ablated_r2, ablated_nll=ablated_nll,
             delta_r2=delta_r2, delta_nll=delta_nll,
             baseline_r2=baseline_r2, baseline_nll=baseline_nll,
             task=args.task, seed=args.seed, n_calib=n_calib)
    print(f"Wrote {out_npz}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    layers = np.arange(N_LAYERS)

    ax = axes[0]
    ax.plot(layers, ablated_r2, "o-", color="C0", lw=2, label="ablated R²")
    ax.axhline(baseline_r2, color="grey", ls="--", lw=1,
               label=f"distract baseline ({baseline_r2:.3f})")
    ax.set_xlabel("Ablation layer k")
    ax.set_ylabel("Mean probe R²")
    ax.set_title("A. Probe R² after projecting out d[k]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(layers, delta_r2, "o-", color="C0", lw=2, label="ΔR² (ablated − baseline)")
    ax.plot(layers, delta_nll, "s-", color="C3", lw=2, label="ΔNLL (ablated − baseline)")
    ax.axhline(0, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("Ablation layer k")
    ax.set_ylabel("Effect of projecting out distractor direction")
    ax.set_title("B. Ablation effect vs layer")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"CAA-style distractor ablation: {args.task} | seed={args.seed}")
    fig.tight_layout()
    out_png = out_dir / f"{args.task}_s{args.seed}.png"
    fig.savefig(str(out_png), dpi=140)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
