"""Cross-θ linear probe of PFN-NPE per-dim encoder embeddings.

PFN-NPE's `TabPFNEmbedder(label_strategy="per_dim")` runs the encoder once
per θ-dim with that dim as the target column, and concatenates the test-
target-token states into a (n, D * 192) tensor. Slicing axis-1 in chunks
of 192 gives per-dim embeddings e_0, …, e_{D-1} — each is the encoder's
representation of x when "tasked" with predicting θ_i.

This script asks: from e_i, how well can a linear probe predict θ_j?

Output: R²[layer, i, j] tensor + normalized variant R²[i,j] / R²[j,j].
  - Diagonal R²[i, i] reproduces the standard per-dim mean probe.
  - Off-diagonal R²[i, j] (i ≠ j) measures whether θ_j-info is preserved
    in the encoder representation when the target column was θ_i.

If R²[i, j] ≈ R²[j, j] for all i: per-dim factorization barely changes
the encoder; e_i ≈ e_j up to a final projection. Joint info available
to the flow.

If R²[i, j] ≪ R²[j, j] when i ≠ j: per-dim factorization genuinely
specializes each embedding to its target. Joint info is destroyed at
the encoder; the flow on top of the concat cannot recover it.

Reads the same `extract_or_load` cache as `layer_linear_probe.py`, so on
tasks that already ran the layer sweep this is essentially zero compute.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.layer_linear_probe import (  # noqa: E402
    N_LAYERS, extract_or_load, fit_mean_probe,
)


def cross_theta_matrix(e_tr: np.ndarray, e_va: np.ndarray,
                       th_tr: np.ndarray, th_va: np.ndarray,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (R²[i, j], α[i, j]) where i = source dim, j = target dim.

    If the embedding width is a multiple of D (per_dim concat layout), each
    source slice e_i is the i-th 192-chunk. If it is NOT divisible (e.g. a
    single shared embedding from label_strategy='random' or 'constant'),
    the same source is used for every i — rows of R² are identical.
    """
    D = th_tr.shape[1]
    width = e_tr.shape[1]
    is_per_dim = (width % D == 0) and (width // D > 1)
    emb_per_dim = width // D if is_per_dim else width

    # Cast to float32 for sklearn (cache stores float16).
    e_tr = e_tr.astype(np.float32, copy=False)
    e_va = e_va.astype(np.float32, copy=False)

    R2 = np.zeros((D, D), dtype=np.float32)
    alpha = np.zeros((D, D), dtype=np.float32)

    for i in range(D):
        if is_per_dim:
            e_tr_i = e_tr[:, i * emb_per_dim:(i + 1) * emb_per_dim]
            e_va_i = e_va[:, i * emb_per_dim:(i + 1) * emb_per_dim]
        else:
            e_tr_i = e_tr
            e_va_i = e_va
        for j in range(D):
            th_tr_j = th_tr[:, j:j + 1]
            th_va_j = th_va[:, j:j + 1]
            best = fit_mean_probe(e_tr_i, th_tr_j, e_va_i, th_va_j)
            R2[i, j] = best["r2"]
            alpha[i, j] = best["alpha"]
    return R2, alpha


def normalize_by_diagonal(R2: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Normalize each column by its diagonal entry.

    R2_norm[i, j] = R2[i, j] / max(R2[j, j], eps).

    A diagonal value at or below ``eps`` is treated as "no signal" and the
    column is zeroed (the off-diagonals are uninterpretable in that regime).
    Diagonal of the result is 1.0 by construction wherever the diagonal of
    the input is positive.
    """
    diag = np.diag(R2).astype(np.float64)
    norm = np.zeros_like(R2, dtype=np.float32)
    for j in range(R2.shape[1]):
        if diag[j] > eps:
            norm[:, j] = R2[:, j] / diag[j]
    return norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--cache-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/probe/cache")
    ap.add_argument("--out-dir",
                    default="pfn_testing/sbi/outputs/layer_ablation/cross_theta")
    ap.add_argument("--layers", type=int, nargs="*",
                    help="Layers to probe (default: all 12).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if output npz exists.")
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"])
    ap.add_argument("--label-strategy", default="per_dim",
                    help="Label strategy for TabPFNEmbedder. Cross-θ matrix "
                         "is only well-defined for 'per_dim'; other strategies "
                         "produce a single shared embedding (broadcast across "
                         "the source axis at probe time).")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts = []
    if args.model_version != "v2":
        suffix_parts.append(f"_m{args.model_version.replace('.', '')}")
    if args.label_strategy != "per_dim":
        suffix_parts.append(f"_ls{args.label_strategy}")
    out_suffix = "".join(suffix_parts)
    out_npz = out_dir / f"{args.task}_s{args.seed}{out_suffix}.npz"
    out_png = out_dir / f"{args.task}_s{args.seed}{out_suffix}.png"
    if out_npz.exists() and not args.force:
        print(f"[skip] {out_npz}")
        return

    layers = args.layers if args.layers else list(range(N_LAYERS))

    R2_per_layer: dict[int, np.ndarray] = {}
    alpha_per_layer: dict[int, np.ndarray] = {}
    D: int | None = None

    print(f"Cross-θ probe: task={args.task} seed={args.seed} "
          f"layers={layers}")
    for k in layers:
        e_tr, e_va, th_tr, th_va = extract_or_load(
            args.task, args.n_train, args.n_val, args.seed, k, cache,
            data=None,
            model_version=args.model_version,
            label_strategy=args.label_strategy,
        )
        if D is None:
            D = th_tr.shape[1]
            print(f"  D={D}, emb width={e_tr.shape[1]}")
        R2, alpha = cross_theta_matrix(e_tr, e_va, th_tr, th_va)
        R2_per_layer[k] = R2
        alpha_per_layer[k] = alpha

        diag = float(np.mean(np.diag(R2)))
        off = R2.copy(); np.fill_diagonal(off, np.nan)
        off_mean = float(np.nanmean(off))
        off_norm = normalize_by_diagonal(R2)
        np.fill_diagonal(off_norm, np.nan)
        off_norm_mean = float(np.nanmean(off_norm))
        print(f"  layer {k:2d}  diag R²={diag:+.4f}  "
              f"off-diag R²={off_mean:+.4f}  "
              f"off-diag (norm)={off_norm_mean:+.4f}")

    assert D is not None
    R2_stack = np.stack([R2_per_layer[k] for k in layers]).astype(np.float32)
    alpha_stack = np.stack([alpha_per_layer[k] for k in layers]).astype(np.float32)
    R2_norm_stack = np.stack(
        [normalize_by_diagonal(R2_per_layer[k]) for k in layers]
    ).astype(np.float32)

    np.savez(
        str(out_npz),
        layers=np.asarray(layers, dtype=np.int32),
        R2=R2_stack, R2_norm=R2_norm_stack, alpha=alpha_stack,
        task=args.task, seed=args.seed,
        n_train=args.n_train, n_val=args.n_val, dim_theta=D,
        model_version=args.model_version, label_strategy=args.label_strategy,
    )
    print(f"\nWrote {out_npz}")

    # ── Per-task summary figure ───────────────────────────────────────────
    import matplotlib.pyplot as plt   # lazy import: keeps headless probes fast
    diag_traj = np.array([np.mean(np.diag(R2_per_layer[k])) for k in layers])
    off_traj = np.array([
        np.nanmean(np.where(np.eye(D, dtype=bool), np.nan, R2_per_layer[k]))
        for k in layers
    ])
    norm_traj = np.array([
        np.nanmean(np.where(np.eye(D, dtype=bool), np.nan,
                            normalize_by_diagonal(R2_per_layer[k])))
        for k in layers
    ])
    best_layer = int(layers[int(np.argmax(diag_traj))])
    R2_at_best = R2_per_layer[best_layer]
    R2_norm_at_best = normalize_by_diagonal(R2_at_best)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(R2_at_best, vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title(f"R²[i, j] @ layer {best_layer}")
    axes[0].set_xlabel("target dim j"); axes[0].set_ylabel("source dim i")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(R2_norm_at_best, vmin=0, vmax=1, cmap="magma")
    axes[1].set_title(f"R²[i,j] / R²[j,j] @ layer {best_layer}")
    axes[1].set_xlabel("target dim j"); axes[1].set_ylabel("source dim i")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].plot(layers, diag_traj, marker="o", label="diag (mean)")
    axes[2].plot(layers, off_traj, marker="s", label="off-diag (mean)")
    axes[2].plot(layers, norm_traj, marker="^", color="C3",
                 label="off-diag / diag (norm)")
    axes[2].axhline(1, color="grey", ls=":", alpha=0.5)
    axes[2].axvline(best_layer, color="grey", ls="--", alpha=0.5,
                    label=f"best layer = {best_layer}")
    axes[2].set_xlabel("encoder layer"); axes[2].set_ylabel("R²")
    axes[2].set_title("Layer trajectory")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    suffix_label = (
        f" | mv={args.model_version} ls={args.label_strategy}"
        if (args.model_version != "v2" or args.label_strategy != "per_dim")
        else ""
    )
    fig.suptitle(f"Cross-θ probe: {args.task} | seed={args.seed}{suffix_label}")
    fig.tight_layout(); fig.savefig(str(out_png), dpi=120)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
