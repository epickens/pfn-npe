"""Aggregate per-seed layer-sweep results into a mean±std C2ST curve.

Reads per-layer per-seed results.npz files produced by tabpfn_npe.run_layer_sweep
and writes a combined npz + png under pfn_testing/sbi/outputs/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_LAYERS = 12


def load_seed(base: Path, n_train: int, seed: int, pooling_suffix: str = "") -> dict:
    """Return dict: layer -> per-observation C2ST array, plus raw baseline."""
    out: dict[int, np.ndarray] = {}
    raw = None
    for k in range(N_LAYERS):
        d = base / f"n{n_train}_per_dim_regressor_layer{k}{pooling_suffix}_s{seed}/results/results.npz"
        if not d.exists():
            raise FileNotFoundError(f"Missing: {d}")
        r = np.load(d, allow_pickle=True)
        out[k] = np.asarray(r["c2st_tabpfn"])
        if raw is None:
            raw = np.asarray(r["c2st_raw"])
    assert raw is not None
    return {"per_layer": out, "raw": raw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="two_moons_distractors")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--pooling-suffix", default="",
                    help="Extra path token after layer{k}; e.g. '_mean' for Task 2 runs.")
    ap.add_argument("--out-prefix", default="pfn_testing/sbi/outputs/layer_ablation/sweep/agg")
    args = ap.parse_args()

    base = Path(f"pfn_testing/sbi/outputs/{args.task}")
    per_seed = [load_seed(base, args.n_train, s, args.pooling_suffix) for s in args.seeds]

    stacked = np.stack(
        [np.concatenate([ps["per_layer"][k] for ps in per_seed]) for k in range(N_LAYERS)]
    )
    means = stacked.mean(axis=1)
    stds = stacked.std(axis=1)

    raw_stack = np.concatenate([ps["raw"] for ps in per_seed])
    raw_mean = float(raw_stack.mean())
    raw_std = float(raw_stack.std())

    best = int(np.argmin(means))
    print(f"Raw baseline: {raw_mean:.4f} ± {raw_std:.4f}")
    print(f"{'layer':>5}  {'C2ST':>8}  {'±std':>8}")
    for k in range(N_LAYERS):
        flag = "  <-- best" if k == best else ""
        print(f"{k:5d}  {means[k]:.4f}  {stds[k]:.4f}{flag}")

    seed_tag = "-".join(map(str, args.seeds))
    out_png = Path(f"{args.out_prefix}{args.pooling_suffix}_s{seed_tag}.png")
    out_npz = Path(f"{args.out_prefix}{args.pooling_suffix}_s{seed_tag}.npz")
    np.savez(
        str(out_npz),
        layers=np.arange(N_LAYERS),
        means=means, stds=stds,
        raw_mean=raw_mean, raw_std=raw_std,
        seeds=np.asarray(args.seeds),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(range(N_LAYERS), means, yerr=stds, marker="o", capsize=4, label="TabPFN-NPE")
    ax.axhline(raw_mean, color="grey", ls="--", label=f"Raw baseline {raw_mean:.3f}")
    ax.fill_between(range(N_LAYERS), raw_mean - raw_std, raw_mean + raw_std, color="grey", alpha=0.2)
    ax.axhline(0.5, color="k", ls=":", label="C2ST = 0.5 (ideal)")
    ax.set_xlabel("TabPFN encoder layer")
    ax.set_ylabel("C2ST (lower = better)")
    ax.set_title(
        f"{args.task} | n={args.n_train} | seeds={args.seeds} | "
        f"pool={args.pooling_suffix.lstrip('_') or 'target'}"
    )
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(str(out_png), dpi=120)
    print(f"Wrote {out_png} and {out_npz}")


if __name__ == "__main__":
    main()
