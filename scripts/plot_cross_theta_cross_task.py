"""Cross-task summary of the cross-θ probe.

Aggregates `cross_theta/{task}_s{seed}.npz` files (default: all tasks/seeds
on disk) and produces two paper-direct figures:

  1. Bar chart of mean off-diagonal-normalized R² (per task, averaged across
     seeds) at the final layer. Sorted ascending — bottom-of-axis tasks have
     the strongest per-dim specialization (off-diag info destroyed); top
     tasks are nearly symmetric (off-diag ≈ diag).

  2. Closing-loop scatter: y = cross-θ off-diag-normalized R², x = NSF's
     joint − marginal C2ST gap (PFN-NPE arm). Tests whether per-dim
     specialization predicts where the flow loses joint structure.

Reads:
  - pfn_testing/sbi/outputs/layer_ablation/cross_theta/*.npz
  - pfn_testing/sbi/outputs/layer_ablation/c2st_summary.csv

Writes:
  - pfn_testing/sbi/outputs/layer_ablation/figures/cross_theta_cross_task.png
  - pfn_testing/sbi/outputs/layer_ablation/figures/cross_theta_vs_joint_gap.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CROSS_THETA_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/cross_theta")
SUMMARY_CSV = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_summary.csv")
FIGURES_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")

NAME_RE = re.compile(
    r"(?P<task>.+?)_s(?P<seed>\d+)(?P<extra>(_m\w+)?(_ls\w+)?)?\.npz$"
)


def load_results(only_default: bool = True) -> pd.DataFrame:
    """Load all cross_theta npzs into a long-form DataFrame.

    Each row: (task, seed, layer, model_version, label_strategy,
               diag_mean, off_diag_mean, off_diag_norm_mean).

    `only_default`: when True, skip files with non-default suffix (m/ls).
    """
    rows = []
    for npz_path in sorted(CROSS_THETA_DIR.glob("*.npz")):
        m = NAME_RE.match(npz_path.name)
        if m is None:
            continue
        if only_default and m.group("extra"):
            continue
        d = np.load(npz_path, allow_pickle=True)
        layers = np.asarray(d["layers"])
        R2 = np.asarray(d["R2"])              # (n_layers, D, D)
        R2_norm = np.asarray(d["R2_norm"])    # (n_layers, D, D)
        D = R2.shape[1]
        eye_mask = np.eye(D, dtype=bool)
        for li, k in enumerate(layers):
            mat = R2[li]
            mat_norm = R2_norm[li]
            diag_mean = float(np.mean(np.diag(mat)))
            off_mean = float(np.nanmean(np.where(eye_mask, np.nan, mat)))
            off_norm_mean = float(
                np.nanmean(np.where(eye_mask, np.nan, mat_norm))
            )
            rows.append({
                "task": m.group("task"),
                "seed": int(m.group("seed")),
                "layer": int(k),
                "extra": m.group("extra") or "",
                "diag_mean": diag_mean,
                "off_diag_mean": off_mean,
                "off_diag_norm_mean": off_norm_mean,
                "D": D,
            })
    return pd.DataFrame(rows)


def joint_minus_marginal(df_c2st: pd.DataFrame, method: str = "nsf",
                         ) -> pd.DataFrame:
    """Per-task joint − marginal C2ST gap for `method`."""
    sub = df_c2st[df_c2st["method"] == method]
    pivot = sub.pivot_table(
        index="task", columns="metric", values="mean", aggfunc="first",
    )
    if not {"joint", "marginal"}.issubset(pivot.columns):
        raise ValueError(
            f"c2st_summary.csv missing 'joint' or 'marginal' for method={method}"
        )
    return (pivot["joint"] - pivot["marginal"]).rename(
        f"joint_minus_marginal_{method}"
    ).to_frame().reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11,
                    help="Layer to summarize at (default 11 = final).")
    ap.add_argument("--method", default="nsf",
                    help="Method whose joint-marginal gap is plotted.")
    args = ap.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results(only_default=True)
    if df.empty:
        raise SystemExit(f"No cross_theta npzs found in {CROSS_THETA_DIR}")
    layer = args.layer
    if layer not in df["layer"].unique():
        raise SystemExit(
            f"Layer {layer} not in available layers {sorted(df['layer'].unique())}"
        )

    layer_df = df[df["layer"] == layer]
    agg = layer_df.groupby("task").agg(
        diag_mean=("diag_mean", "mean"),
        diag_std=("diag_mean", "std"),
        off_diag_mean=("off_diag_mean", "mean"),
        off_diag_std=("off_diag_mean", "std"),
        off_diag_norm_mean=("off_diag_norm_mean", "mean"),
        off_diag_norm_std=("off_diag_norm_mean", "std"),
        n_seeds=("seed", "nunique"),
        D=("D", "first"),
    ).reset_index().sort_values("off_diag_norm_mean")

    # ── Figure 1: cross-task bar of off-diag-normalized R² ───────────────
    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(agg))))
    y = np.arange(len(agg))
    ax.barh(
        y, agg["off_diag_norm_mean"], xerr=agg["off_diag_norm_std"].fillna(0),
        color=["C3" if v < 0.5 else "C2" if v > 0.85 else "C0"
               for v in agg["off_diag_norm_mean"]],
        alpha=0.85, capsize=3,
    )
    ax.axvline(1.0, color="grey", ls=":", alpha=0.5,
               label="off-diag = diag (no specialization)")
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{t} (D={d}, n_seeds={n})"
        for t, d, n in zip(agg["task"], agg["D"], agg["n_seeds"])
    ])
    ax.set_xlabel(
        f"mean off-diagonal-normalized R² @ layer {layer}\n"
        f"(low = strong per-dim specialization, high = symmetric)"
    )
    ax.set_xlim(0, 1.1)
    ax.set_title(f"Cross-θ probe: per-task per-dim specialization")
    ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out_bar = FIGURES_DIR / "cross_theta_cross_task.png"
    fig.savefig(str(out_bar), dpi=120)
    print(f"Wrote {out_bar}")

    # ── Figure 2: closing-loop scatter vs joint-minus-marginal gap ───────
    if not SUMMARY_CSV.exists():
        print(f"[skip closing-loop] {SUMMARY_CSV} not found")
        return
    df_c2st = pd.read_csv(SUMMARY_CSV)
    gap = joint_minus_marginal(df_c2st, method=args.method)
    merged = agg.merge(gap, on="task", how="inner")
    if merged.empty:
        print("[skip closing-loop] no task overlap between cross-θ and c2st")
        return

    gap_col = f"joint_minus_marginal_{args.method}"
    x = merged[gap_col].values
    y = merged["off_diag_norm_mean"].values
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(x, y, s=80, alpha=0.85)
    for _, row in merged.iterrows():
        ax2.annotate(
            row["task"], (row[gap_col], row["off_diag_norm_mean"]),
            fontsize=7, alpha=0.7, xytext=(4, 2), textcoords="offset points",
        )
    if len(x) >= 3:
        try:
            from scipy.stats import pearsonr, spearmanr
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
            ax2.set_title(
                f"Cross-θ specialization vs joint-marginal gap "
                f"(method={args.method})\n"
                f"Pearson r={r_p:+.2f} (p={p_p:.3f}), "
                f"Spearman ρ={r_s:+.2f} (p={p_s:.3f})"
            )
        except ImportError:
            ax2.set_title(
                f"Cross-θ specialization vs joint-marginal gap "
                f"(method={args.method})"
            )
    ax2.set_xlabel(f"{args.method} C2ST: joint − marginal "
                   f"(higher = more joint failure)")
    ax2.set_ylabel(f"mean off-diag-normalized R² @ layer {layer}\n"
                   f"(low = encoder destroyed cross-θ info)")
    ax2.axhline(1.0, color="grey", ls=":", alpha=0.5)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out_scatter = FIGURES_DIR / "cross_theta_vs_joint_gap.png"
    fig2.savefig(str(out_scatter), dpi=120)
    print(f"Wrote {out_scatter}")


if __name__ == "__main__":
    main()
