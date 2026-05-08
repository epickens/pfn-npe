"""Causal localization of distractor filtering via activation patching.

For each encoder layer k, replace the recipient run's layer-k output with
the source run's layer-k output. Measure probe performance on the patched
final-layer embedding. The layer at which recovery saturates is where the
clean signal is causally sufficient for downstream computation.

Source: xs_val with distractor positions zeroed. Same context.
Recipient: standard xs_val (with distractors).

Probes are trained once on distract train embeddings (layer 11) and held
fixed across patched evaluations — we test whether the probe, which was
calibrated on the distract distribution, reads the patched representation.

Patching respects the per_dim label strategy: each θ-dim runs a separate
forward pass (with its own context labels), so source activations and the
patch hook are per-(layer, θ-dim). Final embeddings are concatenated
across θ-dims the same way ``TabPFNEmbedder.transform`` does.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402
from scripts.layer_linear_probe import (  # noqa: E402
    EPS, LOG_2PI, N_LAYERS, fit_mean_probe, fit_variance_probe,
)


def distractor_position_mask(task_name: str) -> np.ndarray:
    """Boolean mask of length ``dim_x`` — True where a distractor feature lives.

    Handles two cases:
      - Our ``DistractorTask`` wrapper (``_permutation`` and ``base_dim_x`` attributes).
      - sbibm's built-in ``slcp_distractors`` (uses a permutation file on disk
        and ``distractors=True``; base slcp has 8 real dims, rest are GMM noise).
    """
    task = get_task(task_name)

    # Our DistractorTask wrapper
    if hasattr(task, "_permutation") and hasattr(task, "base_dim_x"):
        perm = task._permutation
        return np.asarray(perm) >= task.base_dim_x

    # sbibm's slcp_distractors: permutation saved as a file
    if getattr(task, "distractors", False) and hasattr(task, "path"):
        import torch
        perm_file = task.path / "files" / "permutation_idx.torch"
        if perm_file.exists():
            perm = torch.load(str(perm_file), weights_only=False)
            # base slcp has 8 real features; noise appended after
            base_real = 8 if "slcp" in task_name else None
            if base_real is None:
                raise ValueError(
                    f"Unknown base-real dim for sbibm task {task_name}"
                )
            return (perm.numpy() >= base_real)

    raise ValueError(
        f"Task {task_name} doesn't have a distractor-position mask we can compute."
    )


def _register_extraction_hooks(layers, layer_indices, captured: dict):
    """Register forward hooks that store each layer's output into ``captured``.

    Uses ``.clone()`` on top of ``.detach()``: TabPFN may reuse the output
    buffer across layers for memory efficiency, so without cloning all
    ``captured[k]`` entries can end up aliasing the same final-layer output.
    """
    handles = []
    for k in layer_indices:
        def make_hook(k_fixed):
            def hook(module, input, output):
                captured[k_fixed] = output.detach().clone()
            return hook
        handles.append(layers[k].register_forward_hook(make_hook(k)))
    return handles


def _pool_target(layer_out, n_ctx: int) -> np.ndarray:
    """Target-token pooling over test rows.

    layer_out shape: (1, n_ctx + n_test, n_features + 1, emb_dim).
    Returns (n_test, emb_dim).
    """
    test_emb = layer_out[:, n_ctx:, -1]          # (1, n_test, emb_dim)
    emb = test_emb.squeeze(0).cpu().numpy()
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb


def patched_transform_all_layers(
    embedder: TabPFNEmbedder,
    xs_source: np.ndarray,
    xs_recipient: np.ndarray,
    patch_layers: list[int],
    final_layer: int = 11,
) -> dict[int, np.ndarray]:
    """Run the full patching sweep, returning patched embeddings per k.

    For each θ-dim:
      1. Re-fit the TabPFN classifier with that dim's labels.
      2. One forward pass on ``xs_source`` with hooks at every ``patch_layers``
         entry — captures source activations at all target layers.
      3. For each k in ``patch_layers``, one forward pass on ``xs_recipient``
         with two hooks: a patch hook at layer k (replaces output with source's)
         and an extraction hook at ``final_layer`` (captures the patched
         final-layer output).
    Results are concatenated across θ-dims to match ``transform``'s output.

    Returns:
        dict k -> (n_test, dim_theta * emb_dim) patched embedding.
    """
    assert embedder.clf is not None
    assert embedder._per_dim_labels is not None, \
        "This script assumes label_strategy='per_dim'."
    per_dim_labels = embedder._per_dim_labels
    x_context = embedder._x_context
    n_ctx = embedder._n_context

    # patched_parts[k][d] = (n_test, emb_dim)
    patched_parts: dict[int, list] = {k: [None] * len(per_dim_labels) for k in patch_layers}

    for d, labels in enumerate(per_dim_labels):
        embedder.clf.fit(x_context, labels)
        embedder._apply_encoder_checkpoint()
        # TabPFN rebuilds the transformer inside clf.fit; refetch the layer list.
        layers = embedder._get_transformer_model().transformer_encoder.layers

        # Extract source activations at each target layer in a single forward pass
        source_cap: dict = {}
        ext_handles = _register_extraction_hooks(layers, patch_layers, source_cap)
        try:
            embedder.clf.get_embeddings(xs_source, data_source="test")
        finally:
            for h in ext_handles:
                h.remove()

        # Patched forward per k on xs_recipient.
        # Only the test-row portion of layer-k's activation is replaced with
        # source's test rows. Context rows stay as-is. This lets recipient's
        # distract-context still cross-attend to patched-clean test rows,
        # producing variation across k rather than replacing the full forward
        # pass with the source activation.
        for k in patch_layers:
            src_act = source_cap[k]
            fin_cap: dict = {}

            def patch_hook(module, input, output, src=src_act, n_ctx_=n_ctx):
                new_out = output.clone()
                new_out[:, n_ctx_:] = src[:, n_ctx_:]
                return new_out

            def extract_hook(module, input, output):
                fin_cap["out"] = output.detach()

            h1 = layers[k].register_forward_hook(patch_hook)
            h2 = layers[final_layer].register_forward_hook(extract_hook)
            try:
                embedder.clf.get_embeddings(xs_recipient, data_source="test")
            finally:
                h1.remove()
                h2.remove()

            patched_parts[k][d] = _pool_target(fin_cap["out"], n_ctx)

        # Release the per-θ-dim source tensors
        source_cap.clear()

    return {k: np.concatenate(patched_parts[k], axis=1) for k in patch_layers}


def apply_probes(emb, thetas, mu_model, lv_model) -> tuple[float, float]:
    """Return (R², NLL) for a trained mean probe + log-variance probe on ``emb``."""
    mu = mu_model.predict(emb)
    r2 = float(r2_score(thetas, mu, multioutput="uniform_average"))
    r_val = (thetas - mu) ** 2
    log_var = lv_model.predict(emb)
    sigma_sq = np.maximum(np.exp(log_var), EPS)
    nll = float(np.mean(0.5 * (r_val / sigma_sq + np.log(sigma_sq) + LOG_2PI)))
    return r2, nll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="two_moons_distractors")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/probe/cache")
    ap.add_argument("--out-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/patch")
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

    xs_val_clean = xs_val.copy()
    xs_val_clean[:, noise_mask] = 0.0

    # Train probes at layer 11 using cached distract embeddings where available.
    key = cache / f"{args.task}_n{args.n_train}_layer11_s{args.seed}.npz"
    if key.exists():
        print(f"Loading cached layer-11 distract embeddings from {key}")
        d = np.load(key)
        e_tr, e_va = d["emb_train"], d["emb_val"]
    else:
        print("Cache miss on layer-11 embeddings; extracting...")
        emb11 = TabPFNEmbedder(
            context_size=1000, seed=args.seed,
            label_strategy="per_dim", layer=11, model_type="regressor",
        )
        emb11.fit(xs_train, thetas=th_tr)
        e_tr = emb11.transform(xs_train)
        e_va = emb11.transform(xs_val)
        cache.mkdir(parents=True, exist_ok=True)
        np.savez(str(key),
                 emb_train=e_tr, emb_val=e_va,
                 thetas_train=th_tr, thetas_val=th_va)

    print("Fitting probes on distract train embeddings (layer 11)...")
    mean_best = fit_mean_probe(e_tr, th_tr, e_va, th_va)
    var_best = fit_variance_probe(e_tr, th_tr, e_va, th_va,
                                  alpha_mu=mean_best["alpha"])
    mean_model = mean_best["model"]
    lv_model = var_best["log_var_model"]
    mu_probe_model = var_best["mu_model"]   # used as fixed μ head for NLL
    recipient_r2 = mean_best["r2"]
    recipient_nll = var_best["nll"]
    print(f"  recipient baseline: R²={recipient_r2:+.4f}  NLL={recipient_nll:+.4f}")

    # Fit a fresh embedder (default path) for the patching sweep.
    print("Fitting embedder for patching (context = distract xs_train)...")
    embedder = TabPFNEmbedder(
        context_size=1000, seed=args.seed,
        label_strategy="per_dim", layer=None, model_type="regressor",
    )
    embedder.fit(xs_train, thetas=th_tr)

    print("Source baseline: transforming xs_val_clean (same context)...")
    source_emb = embedder.transform(xs_val_clean)
    source_r2, source_nll = apply_probes(source_emb, th_va, mu_probe_model, lv_model)
    # Also compute R² under mean probe's ridge (not joint μ) for reference
    source_r2_ridge = float(r2_score(
        th_va, mean_model.predict(source_emb), multioutput="uniform_average",
    ))
    print(f"  source baseline: R²={source_r2:+.4f} (ridge R²={source_r2_ridge:+.4f})  NLL={source_nll:+.4f}")

    print("Running patched sweep across 12 layers...")
    patch_layers = list(range(N_LAYERS))
    patched_embs = patched_transform_all_layers(
        embedder, xs_val_clean, xs_val, patch_layers,
    )

    patched_r2 = np.zeros(N_LAYERS)
    patched_nll = np.zeros(N_LAYERS)
    for k in patch_layers:
        r2, nll = apply_probes(patched_embs[k], th_va, mu_probe_model, lv_model)
        patched_r2[k] = r2
        patched_nll[k] = nll
        # Recovery fractions: positive-better for both metrics
        denom_r2 = max(1e-8, source_r2 - recipient_r2)
        denom_nll = max(1e-8, recipient_nll - source_nll)
        rec_r2 = (r2 - recipient_r2) / denom_r2
        rec_nll = (recipient_nll - nll) / denom_nll
        print(f"  layer {k:2d}  R²={r2:+.4f} (rec={rec_r2:+.3f})  "
              f"NLL={nll:+.4f} (rec={rec_nll:+.3f})")

    rec_r2_arr = (patched_r2 - recipient_r2) / max(1e-8, source_r2 - recipient_r2)
    rec_nll_arr = (recipient_nll - patched_nll) / max(1e-8, recipient_nll - source_nll)

    out_npz = out_dir / f"{args.task}_s{args.seed}.npz"
    np.savez(str(out_npz),
             layers=np.arange(N_LAYERS),
             patched_r2=patched_r2, patched_nll=patched_nll,
             recovery_r2=rec_r2_arr, recovery_nll=rec_nll_arr,
             recipient_r2=recipient_r2, recipient_nll=recipient_nll,
             source_r2=source_r2, source_nll=source_nll,
             task=args.task, seed=args.seed)
    print(f"Wrote {out_npz}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    layers = np.arange(N_LAYERS)

    ax = axes[0]
    ax.plot(layers, patched_r2, "o-", color="C0", lw=2, label="patched R²")
    ax.axhline(recipient_r2, color="grey", ls="--", lw=1,
               label=f"recipient baseline ({recipient_r2:.3f})")
    ax.axhline(source_r2, color="C2", ls="--", lw=1,
               label=f"source baseline ({source_r2:.3f})")
    ax.set_xlabel("Patch layer k")
    ax.set_ylabel("Mean probe R²")
    ax.set_title("A. Patched mean probe R² vs patch layer")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(layers, rec_r2_arr, "o-", color="C0", lw=2, label="R² recovery")
    ax.plot(layers, rec_nll_arr, "s-", color="C3", lw=2, label="NLL recovery")
    ax.axhline(0, color="grey", ls=":", lw=0.7)
    ax.axhline(1, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("Patch layer k")
    ax.set_ylabel("Recovery fraction (0 = recipient, 1 = source)")
    ax.set_title("B. Recovery fraction vs patch layer")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"Clean→distract patching: {args.task} | seed={args.seed}")
    fig.tight_layout()
    out_png = out_dir / f"{args.task}_s{args.seed}.png"
    fig.savefig(str(out_png), dpi=140)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
