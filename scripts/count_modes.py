"""Mode-count diagnostic on 2D-θ posteriors: flow samples vs MCMC samples.

For each sbibm reference observation, fit 2D KDEs separately on the flow
samples (loaded from flow_vs_quantile/{task}_s{seed}.npz) and the MCMC
reference samples (from sbibm). Evaluate each KDE on a 100×100 grid,
find local maxima above THRESH × peak via scipy.ndimage.maximum_filter,
and count them. Per-obs mode counts surface "mode collapse" — flow drops
a mode that MCMC has — independently of marginal calibration.

Outputs a per-task per-seed npz + a 2×n_ref figure (top row MCMC, bottom
row flow, each panel: samples scatter + KDE contours + ✕ at detected modes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task  # noqa: E402

CMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/mode_count")

DEFAULT_GRID = 100
DEFAULT_THRESH = 0.20
DEFAULT_NEIGHBORHOOD = 15        # ~15% of grid extent; merges crescent micro-peaks
DEFAULT_MARGIN = 0.10
DEFAULT_BW_FACTOR = 1.5          # multiplier on Scott-rule bandwidth; smoother KDE


def find_modes_2d(
    samples: np.ndarray,
    grid_size: int = DEFAULT_GRID,
    thresh: float = DEFAULT_THRESH,
    neighborhood: int = DEFAULT_NEIGHBORHOOD,
    margin: float = DEFAULT_MARGIN,
    bw_factor: float = DEFAULT_BW_FACTOR,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mode_xy, density_grid, xs, ys).

    `mode_xy` is shape (n_modes, 2), each row a (θ_0, θ_1) location of a
    detected local-max above thresh × peak density.
    """
    if bounds is None:
        x_min, y_min = samples.min(axis=0)
        x_max, y_max = samples.max(axis=0)
        x_pad = margin * (x_max - x_min + 1e-9)
        y_pad = margin * (y_max - y_min + 1e-9)
        bounds = (x_min - x_pad, x_max + x_pad,
                  y_min - y_pad, y_max + y_pad)
    xs = np.linspace(bounds[0], bounds[1], grid_size)
    ys = np.linspace(bounds[2], bounds[3], grid_size)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid_pts = np.stack([X.ravel(), Y.ravel()])
    kde = gaussian_kde(samples.T)
    kde.set_bandwidth(bw_method=kde.factor * bw_factor)
    Z = kde(grid_pts).reshape(grid_size, grid_size)

    Z_max = Z.max()
    if Z_max <= 0:
        return np.zeros((0, 2)), Z, xs, ys
    abs_thresh = thresh * Z_max
    local_max = (Z == maximum_filter(Z, size=neighborhood)) & (Z >= abs_thresh)
    coords = np.argwhere(local_max)                       # (n_modes, 2) as (row, col) = (y, x)
    if coords.size == 0:
        return np.zeros((0, 2)), Z, xs, ys
    xy = np.stack([xs[coords[:, 1]], ys[coords[:, 0]]], axis=-1)
    return xy, Z, xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--n-mcmc", type=int, default=10000)
    ap.add_argument("--grid-size", type=int, default=DEFAULT_GRID)
    ap.add_argument("--thresh", type=float, default=DEFAULT_THRESH)
    ap.add_argument("--neighborhood", type=int, default=DEFAULT_NEIGHBORHOOD)
    ap.add_argument("--flow-type", default=None,
                    help="Tag for non-NSF estimators (e.g. 'fmpe', "
                         "'mixture_nsf_K2'). None = vanilla NSF (no suffix).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.flow_type}" if args.flow_type else ""
    p = CMP_DIR / f"{args.task}_s{args.seed}{suffix}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}; run compare_flow_vs_quantile first.")
    cmp_data = dict(np.load(p, allow_pickle=True))
    flow_samples_all = cmp_data["flow_samples"]            # (n_ref, n_samples, 2)
    if flow_samples_all.shape[2] != 2:
        raise SystemExit(
            f"Mode count only supported for 2D θ; this task has dim_theta="
            f"{flow_samples_all.shape[2]}"
        )

    task = get_task(args.task)
    print(f"task={args.task} seed={args.seed}")

    n_ref = min(args.n_ref, flow_samples_all.shape[0])
    flow_modes_per_obs: list[np.ndarray] = []
    mcmc_modes_per_obs: list[np.ndarray] = []
    flow_count = np.zeros(n_ref, dtype=int)
    mcmc_count = np.zeros(n_ref, dtype=int)

    fig, axes = plt.subplots(2, n_ref, figsize=(2.6 * n_ref, 5.6),
                             sharex=False, sharey=False)
    if n_ref == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_ref):
        flow = flow_samples_all[i]
        mcmc = task.get_reference_posterior_samples(num_observation=i + 1).numpy()
        if len(mcmc) > args.n_mcmc:
            mcmc = mcmc[: args.n_mcmc]

        # Use union bounds so both KDEs share the same grid extent.
        all_pts = np.concatenate([flow, mcmc], axis=0)
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        x_pad = DEFAULT_MARGIN * (x_max - x_min + 1e-9)
        y_pad = DEFAULT_MARGIN * (y_max - y_min + 1e-9)
        bounds = (x_min - x_pad, x_max + x_pad,
                  y_min - y_pad, y_max + y_pad)

        modes_mcmc, Z_mcmc, xs, ys = find_modes_2d(
            mcmc, args.grid_size, args.thresh, args.neighborhood,
            bounds=bounds,
        )
        modes_flow, Z_flow, _, _ = find_modes_2d(
            flow, args.grid_size, args.thresh, args.neighborhood,
            bounds=bounds,
        )
        mcmc_count[i] = len(modes_mcmc)
        flow_count[i] = len(modes_flow)
        mcmc_modes_per_obs.append(modes_mcmc)
        flow_modes_per_obs.append(modes_flow)

        for row, (samples, modes, Z, color, name) in enumerate([
            (mcmc, modes_mcmc, Z_mcmc, "C0", "MCMC"),
            (flow, modes_flow, Z_flow, "C3", "flow"),
        ]):
            ax = axes[row, i]
            ax.scatter(samples[:, 0], samples[:, 1], s=4, color=color,
                       alpha=0.20, lw=0)
            ax.contour(xs, ys, Z, levels=6, linewidths=0.6, colors="k",
                       alpha=0.5)
            if len(modes) > 0:
                ax.scatter(modes[:, 0], modes[:, 1], marker="x",
                           s=80, lw=2, color="orange", zorder=4)
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"obs {i+1}\nMCMC: {len(modes_mcmc)} mode(s)",
                             fontsize=8)
            else:
                ax.set_xlabel(f"flow: {len(modes_flow)} mode(s)", fontsize=8)
        print(f"  obs {i+1}: MCMC modes={mcmc_count[i]}, flow modes={flow_count[i]}")

    n_match = int((flow_count == mcmc_count).sum())
    n_collapse = int((flow_count < mcmc_count).sum())
    n_overcount = int((flow_count > mcmc_count).sum())
    print(f"\n{n_match}/{n_ref} obs match, {n_collapse} flow collapse, "
          f"{n_overcount} flow over-count")

    fig.suptitle(
        f"Mode count: {args.task} | seed={args.seed} | "
        f"match {n_match}/{n_ref}, collapse {n_collapse}, over {n_overcount}",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = OUT_DIR / f"{args.task}_s{args.seed}{suffix}.png"
    fig.savefig(str(out_png), dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")

    out_npz = OUT_DIR / f"{args.task}_s{args.seed}{suffix}.npz"
    np.savez(
        str(out_npz),
        flow_count=flow_count, mcmc_count=mcmc_count,
        n_match=n_match, n_collapse=n_collapse, n_overcount=n_overcount,
        task=args.task, seed=args.seed, flow_type=args.flow_type or "nsf",
    )
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
