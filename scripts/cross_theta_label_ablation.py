"""Label-strategy ablation for the cross-θ probe.

For each label_strategy ∈ {per_dim, random, constant}:
  - Extract layer-by-layer encoder embeddings from a single (task, seed) cell.
  - Run the cross-θ matrix probe.

Saves a side-by-side comparison plot. The per-dim strategy should be the only
setting where off-diagonal-normalized R² falls below the diagonal after early
layers, because the random and constant strategies do not provide a
parameter-indexed signal during extraction. Their cross-θ matrices are
broadcast, so off-diagonal and diagonal entries should match by construction.

Default: slcp seed 42 (D=5, dim_x=8). Override via flags.

Output:
  - pfn_testing/sbi/outputs/layer_ablation/cross_theta/{task}_s{seed}_ls{strategy}.npz
    (one per non-default strategy)
  - pfn_testing/sbi/outputs/layer_ablation/figures/cross_theta_label_ablation_{task}_s{seed}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cross_theta_probe import (  # noqa: E402
    cross_theta_matrix, normalize_by_diagonal,
)
from scripts.layer_linear_probe import N_LAYERS, extract_or_load  # noqa: E402

CROSS_THETA_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/cross_theta")
CACHE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/probe/cache")
FIGURES_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")

STRATEGIES = ("per_dim", "random", "constant")


def run_one_strategy(task: str, seed: int, n_train: int, n_val: int,
                     strategy: str) -> dict:
    """Layer-by-layer cross-θ matrix for a single label_strategy."""
    R2_per_layer: list[np.ndarray] = []
    for k in range(N_LAYERS):
        e_tr, e_va, th_tr, th_va = extract_or_load(
            task, n_train, n_val, seed, k, CACHE_DIR,
            data=None, label_strategy=strategy,
        )
        R2, _ = cross_theta_matrix(e_tr, e_va, th_tr, th_va)
        R2_per_layer.append(R2)

    D = R2_per_layer[0].shape[0]
    R2_stack = np.stack(R2_per_layer)
    R2_norm_stack = np.stack(
        [normalize_by_diagonal(R2_per_layer[k]) for k in range(N_LAYERS)]
    )
    eye_mask = np.eye(D, dtype=bool)
    diag_traj = np.array([np.mean(np.diag(R2_per_layer[k]))
                          for k in range(N_LAYERS)])
    off_traj = np.array([
        np.nanmean(np.where(eye_mask, np.nan, R2_per_layer[k]))
        for k in range(N_LAYERS)
    ])
    norm_traj = np.array([
        np.nanmean(np.where(eye_mask, np.nan, R2_norm_stack[k]))
        for k in range(N_LAYERS)
    ])
    return {
        "strategy": strategy, "D": D,
        "R2": R2_stack, "R2_norm": R2_norm_stack,
        "diag_traj": diag_traj, "off_traj": off_traj,
        "norm_traj": norm_traj,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--final-layer", type=int, default=11)
    args = ap.parse_args()

    CROSS_THETA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for strategy in STRATEGIES:
        print(f"\n=== Running label_strategy={strategy} ===")
        res = run_one_strategy(
            args.task, args.seed, args.n_train, args.n_val, strategy,
        )
        results[strategy] = res
        # Save npz for non-default strategies (per_dim already saved by main probe).
        if strategy != "per_dim":
            out_npz = CROSS_THETA_DIR / (
                f"{args.task}_s{args.seed}_ls{strategy}.npz"
            )
            np.savez(
                str(out_npz),
                layers=np.arange(N_LAYERS),
                R2=res["R2"].astype(np.float32),
                R2_norm=res["R2_norm"].astype(np.float32),
                task=args.task, seed=args.seed,
                model_version="v2", label_strategy=strategy,
            )
            print(f"Wrote {out_npz}")

    # ── Comparison figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, len(STRATEGIES), figsize=(15, 9))
    L = args.final_layer
    for col, strategy in enumerate(STRATEGIES):
        res = results[strategy]
        D = res["D"]
        R2_norm = res["R2_norm"][L]

        # Top row: normalized cross-θ heatmap at the final layer.
        ax = axes[0, col]
        im = ax.imshow(R2_norm, vmin=0, vmax=1.2, cmap="magma")
        ax.set_title(f"{strategy}\nR²[i,j] / R²[j,j]  (layer {L})")
        ax.set_xlabel("target dim j")
        if col == 0:
            ax.set_ylabel("source dim i")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Bottom row: layer trajectory of diag, off-diag, normalized.
        ax2 = axes[1, col]
        ax2.plot(range(N_LAYERS), res["diag_traj"], marker="o",
                 label="diag (mean)")
        ax2.plot(range(N_LAYERS), res["off_traj"], marker="s",
                 label="off-diag (mean)")
        ax2.plot(range(N_LAYERS), res["norm_traj"], marker="^", color="C3",
                 label="off-diag / diag (norm)")
        ax2.axhline(1.0, color="grey", ls=":", alpha=0.5)
        ax2.axvline(L, color="grey", ls="--", alpha=0.5)
        ax2.set_xlabel("encoder layer")
        if col == 0:
            ax2.set_ylabel("R²")
        ax2.set_title("Layer trajectory")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.25)

    fig.suptitle(
        f"Label-strategy ablation: cross-θ probe on {args.task} seed={args.seed}\n"
        f"(per_dim is the only strategy that yields a true per-dim source "
        f"representation; others share one embedding across i)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = FIGURES_DIR / (
        f"cross_theta_label_ablation_{args.task}_s{args.seed}.png"
    )
    fig.savefig(str(out_png), dpi=120)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
