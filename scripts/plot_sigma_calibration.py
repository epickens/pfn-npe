"""§6 calibration figure: diag(Σ) for each copula type vs the ideal of 1.

Sklar's decomposition fits a Gaussian copula on z = Φ⁻¹(F̂_d(θ_d|x)). When
F̂_d is *exact* (linear-quantile probe + piecewise-linear F̂), the
uniformization is on (0,1), Φ⁻¹∘F̂ is N(0,1), and diag(Σ) ≈ 1.
When F̂_d is approximated by a 1D NSF, calibration error inflates the
diagonal — diag(Σ) > 1.

The figure makes that argument visually:
  - Cop-Gauss + Cop-Neural (linear-quantile marginals) hug y=1
  - Cop-FullN (1D NSF marginals) bulges away from 1

Usage:
  uv run python scripts/plot_sigma_calibration.py
  uv run python scripts/plot_sigma_calibration.py --out figures/sigma_diag.pdf
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FVQ_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
DEFAULT_OUT = Path(
    "pfn_testing/sbi/outputs/layer_ablation/figures/sigma_diag_calibration.png"
)

NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)_(?P<method>copula.*)$")

COPULA_LABELS = {
    "copula":              "Gauss-Cop\n(linear marg)",
    "copula_neural":       "Neural-Cop\n(linear marg)",
    "copula_fully_neural": "Fully-Neural-Cop\n(1D NSF marg)",
}
COPULA_ORDER = ["copula", "copula_neural", "copula_fully_neural"]
COPULA_COLORS = {
    "copula":              "#2c7bb6",  # blue
    "copula_neural":       "#5cb85c",  # green
    "copula_fully_neural": "#d7191c",  # red — the under-calibrated one
}


def parse_name(stem: str) -> tuple[str, int, str] | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    return m.group("task"), int(m.group("seed")), m.group("method")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Restrict tasks. Default: all with copula data on disk.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--width", type=float, default=12.0)
    ap.add_argument("--height", type=float, default=4.0)
    args = ap.parse_args()

    # ── Collect diag(Σ) per (task, copula_type) ──────────────────────────
    # For each task, for each copula type, pool all (seed, dim) diag values.
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for p in sorted(FVQ_DIR.glob("*_copula*.npz")):
        parsed = parse_name(p.stem)
        if not parsed:
            continue
        task, seed, method = parsed
        if args.tasks is not None and task not in args.tasks:
            continue
        if method not in COPULA_LABELS:
            continue
        try:
            d = np.load(p, allow_pickle=True)
            sigma = d["copula_sigma"]
        except (KeyError, OSError):
            continue
        diag = np.diag(np.asarray(sigma))
        grouped[task][method].extend(diag.tolist())

    if not grouped:
        sys.exit("No *_copula*.npz files with copula_sigma found.")

    tasks = sorted(grouped.keys())
    print(f"Plotting {len(tasks)} tasks × {len(COPULA_ORDER)} copula types")
    for t in tasks:
        for m in COPULA_ORDER:
            n = len(grouped[t].get(m, []))
            print(f"  {t:<32} {m:<22} n={n}")

    # ── Plot: one subplot per task; jittered points + box per copula ─────
    n_tasks = len(tasks)
    fig, axes = plt.subplots(
        1, n_tasks, figsize=(args.width, args.height),
        sharey=True, squeeze=False,
    )
    rng = np.random.default_rng(0)

    for ax, task in zip(axes[0], tasks):
        for j, method in enumerate(COPULA_ORDER):
            vals = np.array(grouped[task].get(method, []))
            if vals.size == 0:
                continue
            x_jit = j + (rng.random(vals.size) - 0.5) * 0.3
            ax.scatter(
                x_jit, vals, s=14, alpha=0.5,
                color=COPULA_COLORS[method], edgecolors="none",
            )
            # Box: median + IQR
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            ax.plot([j - 0.25, j + 0.25], [q50, q50],
                    color="k", linewidth=2)
            ax.fill_between([j - 0.2, j + 0.2], q25, q75,
                            color=COPULA_COLORS[method], alpha=0.15)

        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(range(len(COPULA_ORDER)))
        ax.set_xticklabels(
            [COPULA_LABELS[m] for m in COPULA_ORDER], fontsize=8,
        )
        ax.set_title(task, fontsize=10)
        ax.grid(axis="y", alpha=0.2)

    axes[0, 0].set_ylabel(r"$\mathrm{diag}(\Sigma)$ entries")
    fig.suptitle(
        r"Calibration of Sklar copula marginals: "
        r"$\mathrm{diag}(\Sigma)\!\approx\!1$ ⇔ exact Gaussianization",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    pdf_path = out.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {out}\nWrote {pdf_path}")


if __name__ == "__main__":
    main()
