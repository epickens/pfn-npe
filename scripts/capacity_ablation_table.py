"""Projector-capacity ablation table (P14, SLCP per_dim).

Aggregates joint C2ST across the 3-point projector-capacity sweep on SLCP
(seeds 42, 7, 123) and emits a LaTeX tabular fragment for the appendix.
The point of the table is to show that the joint-vs-marginal C2ST gap is
not an artifact of the linear projector between the per_dim TabPFN
embeddings (5 * 192 = 960-d) and the NSF flow context.

Inputs: pfn_testing/sbi/outputs/slcp/n10000_per_dim_{cfg}_s{seed}/results/results.npz
        where cfg in {joint_linear_64, joint_linear_256, ""} (empty = no reducer).

Usage:
    uv run python scripts/capacity_ablation_table.py
    uv run python scripts/capacity_ablation_table.py \\
        --tex-out results/tables/capacity_ablation.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tex_table import write_tex_tabular  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SLCP_DIR = REPO_ROOT / "pfn_testing/sbi/outputs/slcp"

SEEDS = (42, 7, 123)

# (label, dim_str, dirname-suffix). Empty suffix => no reducer (raw 960-d).
CONFIGS = [
    (r"\texttt{joint\_linear} projector", "64",  "joint_linear_64"),
    (r"\texttt{joint\_linear} projector", "256", "joint_linear_256"),
    (r"no reducer (raw)",                 "960", ""),
]


def load_seed_means(suffix: str) -> np.ndarray:
    """Return array of per-seed C2ST means (one entry per seed)."""
    means = []
    for s in SEEDS:
        if suffix:
            path = SLCP_DIR / f"n10000_per_dim_{suffix}_s{s}" / "results" / "results.npz"
        else:
            path = SLCP_DIR / f"n10000_per_dim_s{s}" / "results" / "results.npz"
        if not path.exists():
            sys.exit(f"Missing: {path}")
        d = np.load(path, allow_pickle=True)
        means.append(float(np.mean(d["c2st_tabpfn"])))
    return np.array(means)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-out", type=Path, default=None,
                    help="Write LaTeX tabular fragment to this path.")
    ap.add_argument("--decimals", type=int, default=3)
    args = ap.parse_args()

    fmt = f".{args.decimals}f"
    rows: list[list[str]] = []

    print(f"{'Configuration':<32} {'d_proj':>7} {'mean':>8} {'std':>8} {'per-seed':>30}")
    print("-" * 90)
    for label, dim_str, suffix in CONFIGS:
        seed_means = load_seed_means(suffix)
        m = float(seed_means.mean())
        s = float(seed_means.std(ddof=0))
        per_seed = ", ".join(f"{x:.3f}" for x in seed_means)
        print(f"{label:<32} {dim_str:>7} {m:>8.4f} {s:>8.4f} {per_seed:>30}")
        cell = f"${m:{fmt}}\\,\\pm\\,{s:.3f}$"
        rows.append([label, dim_str, cell])

    if args.tex_out is not None:
        write_tex_tabular(
            out_path=args.tex_out,
            columns=["Configuration", r"$d_{\text{proj}}$", r"Joint C2ST"],
            rows=rows,
            column_align="lcc",
            source_script="scripts/capacity_ablation_table.py",
        )


if __name__ == "__main__":
    main()
