"""Overlay C2ST (left axis) and linear-probe R² (right axis) per layer."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2st-npz", required=True,
                    help="Aggregated C2ST npz (from aggregate_layer_sweep.py)")
    ap.add_argument("--probe-npz", required=True,
                    help="Probe R² npz (from layer_linear_probe.py)")
    ap.add_argument("--out", default="pfn_testing/sbi/outputs/layer_ablation/figures/c2st_vs_probe.png")
    args = ap.parse_args()

    c = np.load(args.c2st_npz); p = np.load(args.probe_npz)
    layers = c["layers"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(layers, c["means"], yerr=c["stds"], marker="o", color="C0",
                 capsize=4, label="C2ST (↓ = better)")
    ax1.axhline(0.5, color="C0", ls=":", alpha=0.5)
    ax1.set_xlabel("TabPFN encoder layer")
    ax1.set_ylabel("C2ST", color="C0"); ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(layers, p["r2"], marker="s", color="C3", label="Probe R² (↑ = better)")
    ax2.axhline(0, color="C3", ls=":", alpha=0.5)
    ax2.set_ylabel("Ridge val R²", color="C3"); ax2.tick_params(axis="y", labelcolor="C3")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Per-layer: C2ST (flow) vs linear-probe R² (no flow)")
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
