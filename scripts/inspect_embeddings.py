"""Compare raw TabPFN vs contrastive-finetuned embedding distributions.

For a given (task, seed), extracts embeddings with both encoders on the
same train data and reports:

  - Per-dim mean, std summary (boxed across the embedding dimensions)
  - Per-row L2 norm distribution
  - Correlation between raw and v3 embeddings (column-wise)
  - Per-dim histograms (sample of dims) overlaid
  - Train-θ-to-embedding linear-probe R² for both encoders (sanity check
    that the encoder still preserves θ information)

The motivating question: does v3 produce embeddings on a wildly different
scale/geometry than raw TabPFN? If so, that explains why downstream NSF
training (tuned to raw scale) fails on v3.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402

OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")


def fit_embedder(task: str, seed: int, data: dict,
                 encoder_ckpt: str | None) -> tuple[np.ndarray, np.ndarray]:
    emb = TabPFNEmbedder(
        context_size=1000, seed=seed, label_strategy="per_dim",
        layer=None, model_type="regressor",
    )
    emb.fit(data["xs_train"], thetas=data["thetas_train"])
    if encoder_ckpt:
        emb.load_encoder_checkpoint(encoder_ckpt)
    e_tr = emb.transform(data["xs_train"])
    e_va = emb.transform(data["xs_val"])
    return e_tr, e_va


def linear_probe_r2(e_tr: np.ndarray, th_tr: np.ndarray,
                    e_va: np.ndarray, th_va: np.ndarray,
                    alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0)) -> float:
    best = -np.inf
    for a in alphas:
        m = Ridge(alpha=a).fit(e_tr, th_tr)
        r2 = float(r2_score(th_va, m.predict(e_va), multioutput="uniform_average"))
        best = max(best, r2)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--encoder-checkpoint", required=True,
                    help="v3 checkpoint to compare against raw TabPFN")
    ap.add_argument("--encoder-tag", default="v3")
    ap.add_argument("--out-stem",
                    default="emb_compare")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Simulating {args.task} (n={args.n_train})...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)

    print("\nExtracting raw TabPFN embeddings...")
    e_tr_raw, e_va_raw = fit_embedder(args.task, args.seed, data, None)
    print(f"  emb_train shape={e_tr_raw.shape}, dtype={e_tr_raw.dtype}")
    e_tr_raw = e_tr_raw.astype(np.float32, copy=False)
    e_va_raw = e_va_raw.astype(np.float32, copy=False)

    print(f"\nExtracting v3 contrastive embeddings ({args.encoder_tag})...")
    e_tr_v3, e_va_v3 = fit_embedder(args.task, args.seed, data, args.encoder_checkpoint)
    print(f"  emb_train shape={e_tr_v3.shape}, dtype={e_tr_v3.dtype}")
    e_tr_v3 = e_tr_v3.astype(np.float32, copy=False)
    e_va_v3 = e_va_v3.astype(np.float32, copy=False)

    print("\n=== Embedding statistics ===")
    print(f"{'metric':<28} {'raw':>14} {'v3':>14} {'ratio (v3/raw)':>16}")
    for name, fn in [
        ("global mean", lambda x: float(x.mean())),
        ("global std", lambda x: float(x.std())),
        ("global abs mean", lambda x: float(np.abs(x).mean())),
        ("global max abs", lambda x: float(np.abs(x).max())),
        ("per-row L2 mean", lambda x: float(np.linalg.norm(x, axis=1).mean())),
        ("per-row L2 std", lambda x: float(np.linalg.norm(x, axis=1).std())),
        ("per-dim std mean", lambda x: float(x.std(axis=0).mean())),
        ("per-dim std std", lambda x: float(x.std(axis=0).std())),
    ]:
        a = fn(e_tr_raw); b = fn(e_tr_v3)
        ratio = b / a if abs(a) > 1e-12 else float("nan")
        print(f"{name:<28} {a:>14.4f} {b:>14.4f} {ratio:>16.4f}")

    print("\n=== Linear probe (θ-info preserved by encoder) ===")
    r2_raw = linear_probe_r2(e_tr_raw, data["thetas_train"], e_va_raw, data["thetas_val"])
    r2_v3 = linear_probe_r2(e_tr_v3, data["thetas_train"], e_va_v3, data["thetas_val"])
    print(f"  ridge R² on θ | raw : {r2_raw:.4f}")
    print(f"  ridge R² on θ | v3  : {r2_v3:.4f}")
    print(f"  Δ                   : {r2_v3 - r2_raw:+.4f}")

    print("\n=== Per-dim correspondence ===")
    col_corrs = np.array([
        np.corrcoef(e_tr_raw[:, d], e_tr_v3[:, d])[0, 1]
        for d in range(e_tr_raw.shape[1])
    ])
    print(f"  per-dim Pearson r (raw_d, v3_d) — mean: {col_corrs.mean():.3f}, "
          f"std: {col_corrs.std():.3f}")
    print(f"  fraction of dims with |r|>0.5: {(np.abs(col_corrs) > 0.5).mean():.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # A: per-row L2 norm histograms
    ax = axes[0, 0]
    norms_raw = np.linalg.norm(e_tr_raw, axis=1)
    norms_v3 = np.linalg.norm(e_tr_v3, axis=1)
    ax.hist(norms_raw, bins=50, alpha=0.6, label=f"raw (mean={norms_raw.mean():.2f})", color="C0")
    ax.hist(norms_v3, bins=50, alpha=0.6, label=f"v3 (mean={norms_v3.mean():.2f})", color="C3")
    ax.set_xlabel("Per-row L2 norm")
    ax.set_ylabel("Count")
    ax.set_title("A. Embedding magnitudes")
    ax.legend(); ax.grid(True, alpha=0.3)

    # B: per-dim std distribution
    ax = axes[0, 1]
    stds_raw = e_tr_raw.std(axis=0)
    stds_v3 = e_tr_v3.std(axis=0)
    ax.hist(stds_raw, bins=50, alpha=0.6, label=f"raw (mean={stds_raw.mean():.3f})", color="C0")
    ax.hist(stds_v3, bins=50, alpha=0.6, label=f"v3 (mean={stds_v3.mean():.3f})", color="C3")
    ax.set_xlabel("Per-dim std")
    ax.set_ylabel("Count of dims")
    ax.set_title("B. Per-dim variance")
    ax.legend(); ax.grid(True, alpha=0.3)

    # C: column correspondence (raw vs v3 per-dim)
    ax = axes[0, 2]
    ax.hist(col_corrs, bins=50, alpha=0.7, color="C2")
    ax.axvline(0, color="grey", ls=":", alpha=0.5)
    ax.set_xlabel("Pearson r between raw_d and v3_d")
    ax.set_ylabel("Count of dims")
    ax.set_title("C. Per-dim correspondence raw ↔ v3")
    ax.grid(True, alpha=0.3)

    # D: scatter of mean values per dim
    ax = axes[1, 0]
    means_raw = e_tr_raw.mean(axis=0)
    means_v3 = e_tr_v3.mean(axis=0)
    ax.scatter(means_raw, means_v3, s=10, alpha=0.5)
    lim = max(np.abs(means_raw).max(), np.abs(means_v3).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "--", color="grey", alpha=0.5, label="y=x")
    ax.set_xlabel("Raw per-dim mean")
    ax.set_ylabel("v3 per-dim mean")
    ax.set_title("D. Per-dim mean: raw vs v3")
    ax.legend(); ax.grid(True, alpha=0.3)

    # E: scatter of stds per dim
    ax = axes[1, 1]
    ax.scatter(stds_raw, stds_v3, s=10, alpha=0.5)
    lim = max(stds_raw.max(), stds_v3.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="grey", alpha=0.5, label="y=x")
    ax.set_xlabel("Raw per-dim std")
    ax.set_ylabel("v3 per-dim std")
    ax.set_title("E. Per-dim std: raw vs v3")
    ax.legend(); ax.grid(True, alpha=0.3)

    # F: representative dim histograms
    ax = axes[1, 2]
    rng = np.random.default_rng(0)
    pick = rng.choice(e_tr_raw.shape[1], size=4, replace=False)
    for i, d in enumerate(pick):
        ax.hist(e_tr_raw[:, d], bins=50, histtype="step", lw=1.5, color=f"C{i}", ls="-",
                label=f"raw dim {d}" if i == 0 else None)
        ax.hist(e_tr_v3[:, d], bins=50, histtype="step", lw=1.5, color=f"C{i}", ls=":")
    ax.set_xlabel("Embedding value (sample dims)")
    ax.set_ylabel("Count")
    ax.set_title("F. Sample dim distributions (solid=raw, dotted=v3)")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Embedding comparison: raw TabPFN vs {args.encoder_tag} | "
        f"{args.task} | seed={args.seed}",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT_DIR / f"{args.out_stem}_{args.task}_s{args.seed}_{args.encoder_tag}.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
