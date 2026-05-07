"""Overlay two layer-sweep curves (e.g. target vs mean pooling)."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-npz", required=True, help="Aggregated npz for condition A")
    ap.add_argument("--a-label", default="target")
    ap.add_argument("--b-npz", required=True, help="Aggregated npz for condition B")
    ap.add_argument("--b-label", default="mean")
    ap.add_argument("--out", default="pfn_testing/sbi/outputs/layer_ablation/sweep/compare.png")
    args = ap.parse_args()

    a = np.load(args.a_npz); b = np.load(args.b_npz)
    layers = a["layers"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(layers, a["means"], yerr=a["stds"], marker="o", capsize=4,
                label=f"pool={args.a_label}")
    ax.errorbar(layers + 0.15, b["means"], yerr=b["stds"], marker="s", capsize=4,
                label=f"pool={args.b_label}")
    raw_mean = float(a["raw_mean"]); raw_std = float(a["raw_std"])
    ax.axhline(raw_mean, color="grey", ls="--", label=f"Raw baseline {raw_mean:.3f}")
    ax.fill_between(layers, raw_mean - raw_std, raw_mean + raw_std, color="grey", alpha=0.2)
    ax.axhline(0.5, color="k", ls=":", label="C2ST = 0.5")
    ax.set_xlabel("TabPFN encoder layer"); ax.set_ylabel("C2ST (lower = better)")
    ax.set_title("Layer sweep: target vs mean pooling")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
