"""Focused pairwise copula-slice diagnostic for SLCP.

The interaction-order test localized the strongest pure-copula discrepancy to
the (theta_0, theta_1) pair, especially for observation 1. This script compares
that pair directly for PFN-NPE and sbi-NLE:

  * separate-rank copula C2ST on (u0, u1)
  * conditional slices u1 | u0-bin
  * tail-quadrant masses
  * 2D copula histogram total variation
  * Gaussianized radial/angular summaries

It retrains the methods locally because the existing sbi-NLE artifacts save
metrics and quantiles, but not posterior samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from scipy.stats import ks_2samp, norm, wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    sample_posterior,
    train_flow,
)
from pfn_testing.sbi.sbi_baselines import _get_sbi_class, _prior_to_device  # noqa: E402
from pfn_testing.sbi.sbibm_utils import compute_c2st, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import Config, PCAReducer, TabPFNEmbedder  # noqa: E402


def rank_columns(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    ranks = np.empty_like(samples, dtype=np.float64)
    denom = samples.shape[0] + 1.0
    for d in range(samples.shape[1]):
        order = np.argsort(samples[:, d], kind="mergesort")
        ranks[order, d] = (np.arange(samples.shape[0]) + 1.0) / denom
    return ranks


def separate_pair_ranks(
    samples: np.ndarray,
    reference: np.ndarray,
    pair: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(samples), len(reference))
    model_pair = samples[:n, pair]
    ref_pair = reference[:n, pair]
    return rank_columns(model_pair), rank_columns(ref_pair)


def pooled_pair_ranks(
    samples: np.ndarray,
    reference: np.ndarray,
    pair: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(samples), len(reference))
    pooled = np.concatenate([samples[:n, pair], reference[:n, pair]], axis=0)
    ranks = rank_columns(pooled)
    return ranks[:n], ranks[n:]


def hist2d(u: np.ndarray, bins: int) -> np.ndarray:
    h, _, _ = np.histogram2d(
        u[:, 0],
        u[:, 1],
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    h = h.astype(np.float64)
    return h / max(h.sum(), 1.0)


def circular_hist_tv(angle_a: np.ndarray, angle_b: np.ndarray, bins: int) -> float:
    h_a, _ = np.histogram(angle_a, bins=bins, range=(-np.pi, np.pi))
    h_b, _ = np.histogram(angle_b, bins=bins, range=(-np.pi, np.pi))
    p = h_a / max(h_a.sum(), 1)
    q = h_b / max(h_b.sum(), 1)
    return float(0.5 * np.abs(p - q).sum())


def conditional_slice_rows(
    method: str,
    obs: int,
    u_model: np.ndarray,
    u_ref: np.ndarray,
    n_bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            m_mask = (u_model[:, 0] >= lo) & (u_model[:, 0] <= hi)
            r_mask = (u_ref[:, 0] >= lo) & (u_ref[:, 0] <= hi)
        else:
            m_mask = (u_model[:, 0] >= lo) & (u_model[:, 0] < hi)
            r_mask = (u_ref[:, 0] >= lo) & (u_ref[:, 0] < hi)
        m = u_model[m_mask, 1]
        r = u_ref[r_mask, 1]
        ks = ks_2samp(m, r)
        q_model = np.quantile(m, [0.1, 0.5, 0.9])
        q_ref = np.quantile(r, [0.1, 0.5, 0.9])
        rows.append(
            {
                "method": method,
                "obs": obs,
                "bin": b,
                "u0_lo": lo,
                "u0_hi": hi,
                "n_model": int(len(m)),
                "n_ref": int(len(r)),
                "mean_model": float(m.mean()),
                "mean_ref": float(r.mean()),
                "mean_diff": float(m.mean() - r.mean()),
                "var_model": float(m.var()),
                "var_ref": float(r.var()),
                "wasserstein": float(wasserstein_distance(m, r)),
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "q10_diff": float(q_model[0] - q_ref[0]),
                "q50_diff": float(q_model[1] - q_ref[1]),
                "q90_diff": float(q_model[2] - q_ref[2]),
            }
        )
    return rows


def tail_quadrant_rows(
    method: str,
    obs: int,
    u_model: np.ndarray,
    u_ref: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quadrants = {
        "low_low": lambda u, t: (u[:, 0] <= t) & (u[:, 1] <= t),
        "low_high": lambda u, t: (u[:, 0] <= t) & (u[:, 1] >= 1.0 - t),
        "high_low": lambda u, t: (u[:, 0] >= 1.0 - t) & (u[:, 1] <= t),
        "high_high": lambda u, t: (u[:, 0] >= 1.0 - t) & (u[:, 1] >= 1.0 - t),
    }
    for t in thresholds:
        for name, fn in quadrants.items():
            model_mass = float(fn(u_model, t).mean())
            ref_mass = float(fn(u_ref, t).mean())
            rows.append(
                {
                    "method": method,
                    "obs": obs,
                    "threshold": t,
                    "quadrant": name,
                    "model_mass": model_mass,
                    "ref_mass": ref_mass,
                    "mass_diff": model_mass - ref_mass,
                    "abs_mass_diff": abs(model_mass - ref_mass),
                }
            )
    return rows


def method_summary(
    method: str,
    obs: int,
    samples: np.ndarray,
    reference: np.ndarray,
    pair: tuple[int, int],
    hist_bins: int,
    angle_bins: int,
    c2st_folds: int,
    c2st_max_epochs: int,
    seed: int,
) -> dict[str, Any]:
    u_model, u_ref = separate_pair_ranks(samples, reference, pair)
    pooled_model, pooled_ref = pooled_pair_ranks(samples, reference, pair)

    h_model = hist2d(u_model, hist_bins)
    h_ref = hist2d(u_ref, hist_bins)
    diff = h_model - h_ref

    z_model = norm.ppf(np.clip(u_model, 1e-4, 1 - 1e-4))
    z_ref = norm.ppf(np.clip(u_ref, 1e-4, 1 - 1e-4))
    r_model = np.sqrt((z_model**2).sum(axis=1))
    r_ref = np.sqrt((z_ref**2).sum(axis=1))
    a_model = np.arctan2(z_model[:, 1], z_model[:, 0])
    a_ref = np.arctan2(z_ref[:, 1], z_ref[:, 0])

    pair_copula = compute_c2st(
        u_model,
        u_ref,
        n_folds=c2st_folds,
        max_epochs=c2st_max_epochs,
        seed=seed,
    )
    pair_pooled = compute_c2st(
        pooled_model,
        pooled_ref,
        n_folds=c2st_folds,
        max_epochs=c2st_max_epochs,
        seed=seed,
    )
    radius_ks = ks_2samp(r_model, r_ref)
    return {
        "method": method,
        "obs": obs,
        "pair": f"{pair[0]} {pair[1]}",
        "pair_copula_c2st": float(pair_copula),
        "pair_pooled_rank_c2st": float(pair_pooled),
        "hist_tv": float(0.5 * np.abs(diff).sum()),
        "hist_intersection": float(np.minimum(h_model, h_ref).sum()),
        "hist_max_abs_cell": float(np.abs(diff).max()),
        "hist_l2": float(np.sqrt(np.mean(diff**2))),
        "radius_ks": float(radius_ks.statistic),
        "radius_wasserstein": float(wasserstein_distance(r_model, r_ref)),
        "angle_tv": circular_hist_tv(a_model, a_ref, angle_bins),
    }


def aggregate_conditional(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        out.append(
            {
                "method": method,
                "mean_abs_conditional_mean_diff": float(
                    np.mean([abs(r["mean_diff"]) for r in sub])
                ),
                "max_abs_conditional_mean_diff": float(
                    np.max([abs(r["mean_diff"]) for r in sub])
                ),
                "mean_conditional_wasserstein": float(
                    np.mean([r["wasserstein"] for r in sub])
                ),
                "max_conditional_wasserstein": float(
                    np.max([r["wasserstein"] for r in sub])
                ),
                "mean_conditional_ks": float(np.mean([r["ks_stat"] for r in sub])),
                "max_conditional_ks": float(np.max([r["ks_stat"] for r in sub])),
            }
        )
    return out


def aggregate_tail(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        out.append(
            {
                "method": method,
                "mean_abs_tail_mass_diff": float(
                    np.mean([r["abs_mass_diff"] for r in sub])
                ),
                "max_abs_tail_mass_diff": float(
                    np.max([r["abs_mass_diff"] for r in sub])
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_tabpfn_flow(data: dict[str, Any], obs_x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    print("\nTraining PFN-NPE tabpfn_pca flow")
    embedder = TabPFNEmbedder(
        context_size=args.context_size,
        seed=args.seed,
        label_strategy=args.label_strategy,
        model_type=args.model_type,
        device=args.device,
        model_version=args.model_version,
    )
    embedder.fit(data["xs_train"], thetas=data["thetas_train"])
    emb_train = embedder.transform(data["xs_train"])
    emb_val = embedder.transform(data["xs_val"])
    emb_obs = embedder.transform(obs_x.reshape(1, -1))
    reducer = PCAReducer(args.embed_dim)
    ctx_train = reducer.fit_transform(emb_train)
    ctx_val = reducer.transform(emb_val)
    ctx_obs = reducer.transform(emb_obs)[0]

    dim_theta = data["dim_theta"]
    defaults = get_flow_defaults(dim_theta)
    cfg = Config(
        task_name=args.task,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        device=args.device,
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        label_strategy=args.label_strategy,
        model_type=args.model_type,
        flow_type=args.flow_type,
        max_epochs=args.flow_epochs,
        patience=args.flow_patience,
        batch_size=args.flow_batch_size,
        lr=args.flow_lr,
        n_posterior_samples=args.n_posterior_samples,
        n_reference_observations=1,
        skip_raw=True,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    flow = build_flow(
        dim_theta=dim_theta,
        dim_context=ctx_train.shape[1],
        n_transforms=args.n_transforms or defaults["n_transforms"],
        hidden_features=defaults["hidden_features"],
        n_bins=cfg.n_bins,
        flow_type=args.flow_type,
    )
    history = train_flow(
        flow,
        data["thetas_train"],
        ctx_train,
        data["thetas_val"],
        ctx_val,
        cfg,
    )
    return sample_posterior(
        flow,
        ctx_obs,
        history["theta_mean"],
        history["theta_std"],
        args.n_posterior_samples,
    )


def train_raw_flow(data: dict[str, Any], obs_x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    print("\nTraining raw-x NSF flow")
    dim_theta = data["dim_theta"]
    defaults = get_flow_defaults(dim_theta)
    cfg = Config(
        task_name=args.task,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        device=args.device,
        flow_type=args.flow_type,
        max_epochs=args.flow_epochs,
        patience=args.flow_patience,
        batch_size=args.flow_batch_size,
        lr=args.flow_lr,
        n_posterior_samples=args.n_posterior_samples,
        n_reference_observations=1,
        skip_raw=True,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    flow = build_flow(
        dim_theta=dim_theta,
        dim_context=data["dim_x"],
        n_transforms=args.n_transforms or defaults["n_transforms"],
        hidden_features=defaults["hidden_features"],
        n_bins=cfg.n_bins,
        flow_type=args.flow_type,
    )
    history = train_flow(
        flow,
        data["thetas_train"],
        data["xs_train"],
        data["thetas_val"],
        data["xs_val"],
        cfg,
    )
    return sample_posterior(
        flow,
        obs_x,
        history["theta_mean"],
        history["theta_std"],
        args.n_posterior_samples,
    )


def train_sbi_nle(data: dict[str, Any], obs_x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    print("\nTraining sbi-NLE comparator")
    from sbi.neural_nets import likelihood_nn

    if args.nle_sample_with == "vi":
        raise SystemExit(
            "VI sampling in sbi-NLE requires fitting a variational posterior for "
            "the chosen observation. Use --nle-sample-with mcmc for this diagnostic."
        )

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    task = data["task"]
    prior_dist = _prior_to_device(task.get_prior_dist(), device)
    density_builder = likelihood_nn(model=args.nle_density_estimator)
    nle_class = _get_sbi_class("nle")
    inference = nle_class(
        prior=prior_dist,
        density_estimator=density_builder,
        device=device,
    )
    theta_tensor = torch.tensor(data["thetas_train"], dtype=torch.float32)
    x_tensor = torch.tensor(data["xs_train"], dtype=torch.float32)
    inference.append_simulations(theta_tensor, x_tensor)
    train_kwargs: dict[str, Any] = {
        "training_batch_size": args.nle_batch_size,
        "stop_after_epochs": args.nle_stop_after_epochs,
        "learning_rate": args.nle_lr,
    }
    if args.nle_max_num_epochs is not None:
        train_kwargs["max_num_epochs"] = args.nle_max_num_epochs
    inference.train(**train_kwargs)
    build_kwargs: dict[str, Any] = {"sample_with": args.nle_sample_with}
    if args.nle_sample_with == "mcmc":
        build_kwargs["mcmc_method"] = args.nle_mcmc_method
    posterior = inference.build_posterior(**build_kwargs)
    x_obs_tensor = torch.tensor(obs_x, dtype=torch.float32, device=device)
    samples = posterior.sample((args.n_posterior_samples,), x=x_obs_tensor)
    return samples.detach().cpu().numpy()


def plot_diagnostic(
    path: Path,
    methods: list[str],
    ranks: dict[str, tuple[np.ndarray, np.ndarray]],
    conditional_rows: list[dict[str, Any]],
    hist_bins: int,
) -> None:
    fig, axes = plt.subplots(
        len(methods),
        3,
        figsize=(11, 3.2 * len(methods)),
        squeeze=False,
        constrained_layout=True,
    )
    edges = np.linspace(0.0, 1.0, hist_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for i, method in enumerate(methods):
        u_model, u_ref = ranks[method]
        h_model = hist2d(u_model, hist_bins)
        h_ref = hist2d(u_ref, hist_bins)
        diff = h_model - h_ref
        vmax = max(np.abs(diff).max(), 1e-6)
        im = axes[i, 0].imshow(
            diff.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
        )
        axes[i, 0].set_title(f"{method}: model - ref")
        axes[i, 0].set_xlabel("u0")
        axes[i, 0].set_ylabel("u1")
        fig.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)

        axes[i, 1].scatter(
            u_ref[:, 0],
            u_ref[:, 1],
            s=6,
            alpha=0.25,
            label="reference",
            color="#4C78A8",
            rasterized=True,
        )
        axes[i, 1].scatter(
            u_model[:, 0],
            u_model[:, 1],
            s=6,
            alpha=0.25,
            label="model",
            color="#F58518",
            rasterized=True,
        )
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_title("rank copula samples")
        axes[i, 1].set_xlabel("u0")
        axes[i, 1].set_ylabel("u1")
        axes[i, 1].legend(loc="upper right", fontsize=8)

        sub = sorted(
            [r for r in conditional_rows if r["method"] == method],
            key=lambda r: r["bin"],
        )
        axes[i, 2].plot(
            centers[: len(sub)],
            [r["mean_ref"] for r in sub],
            marker="o",
            label="reference",
            color="#4C78A8",
        )
        axes[i, 2].plot(
            centers[: len(sub)],
            [r["mean_model"] for r in sub],
            marker="o",
            label="model",
            color="#F58518",
        )
        axes[i, 2].set_ylim(0, 1)
        axes[i, 2].set_title("E[u1 | u0 bin]")
        axes[i, 2].set_xlabel("u0 bin center")
        axes[i, 2].set_ylabel("mean u1")
        axes[i, 2].grid(alpha=0.25)
        axes[i, 2].legend(loc="upper right", fontsize=8)

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--n-train", type=int, default=3_000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--obs", type=int, default=1)
    ap.add_argument("--pair", default="0,1")
    ap.add_argument("--methods", default="tabpfn,nle", help="Comma list: tabpfn,raw,nle")
    ap.add_argument("--n-posterior-samples", type=int, default=500)
    ap.add_argument("--context-size", type=int, default=1_000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--label-strategy", default="per_dim", choices=["per_dim", "per_dim_mean"])
    ap.add_argument("--model-type", default="regressor", choices=["regressor", "classifier"])
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"])
    ap.add_argument("--flow-type", default="nsf", choices=["nsf", "naf"])
    ap.add_argument("--n-transforms", type=int, default=None)
    ap.add_argument("--flow-epochs", type=int, default=80)
    ap.add_argument("--flow-patience", type=int, default=10)
    ap.add_argument("--flow-batch-size", type=int, default=256)
    ap.add_argument("--flow-lr", type=float, default=5e-4)
    ap.add_argument("--nle-density-estimator", default="maf")
    ap.add_argument("--nle-sample-with", default="mcmc", choices=["mcmc", "vi", "rejection"])
    ap.add_argument("--nle-mcmc-method", default="slice_np_vectorized")
    ap.add_argument("--nle-stop-after-epochs", type=int, default=20)
    ap.add_argument("--nle-max-num-epochs", type=int, default=120)
    ap.add_argument("--nle-batch-size", type=int, default=200)
    ap.add_argument("--nle-lr", type=float, default=5e-4)
    ap.add_argument("--conditional-bins", type=int, default=10)
    ap.add_argument("--hist-bins", type=int, default=12)
    ap.add_argument("--angle-bins", type=int, default=16)
    ap.add_argument("--tail-thresholds", default="0.1,0.2")
    ap.add_argument("--c2st-folds", type=int, default=3)
    ap.add_argument("--c2st-max-epochs", type=int, default=100)
    ap.add_argument("--out-dir", default="pfn_testing/sbi/outputs/focused_pairwise_copula")
    args = ap.parse_args()

    pair_values = tuple(int(x.strip()) for x in args.pair.split(","))
    if len(pair_values) != 2:
        raise SystemExit("--pair must contain exactly two comma-separated indices")
    pair = (pair_values[0], pair_values[1])
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - {"tabpfn", "raw", "nle"}
    if unknown:
        raise SystemExit(f"Unknown methods: {sorted(unknown)}")
    tail_thresholds = [float(x.strip()) for x in args.tail_thresholds.split(",") if x.strip()]

    print(
        f"Simulating {args.task}: n_train={args.n_train}, n_val={args.n_val}, "
        f"seed={args.seed}"
    )
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    task = data["task"]
    obs_x = task.get_observation(num_observation=args.obs).numpy().squeeze(0)
    ref = task.get_reference_posterior_samples(num_observation=args.obs).numpy()
    ref = ref[: args.n_posterior_samples]

    samples_by_method: dict[str, np.ndarray] = {}
    if "tabpfn" in methods:
        samples_by_method[f"tabpfn_pca{args.embed_dim}"] = train_tabpfn_flow(data, obs_x, args)
    if "raw" in methods:
        samples_by_method["raw_x"] = train_raw_flow(data, obs_x, args)
    if "nle" in methods:
        samples_by_method["sbi_nle"] = train_sbi_nle(data, obs_x, args)

    summary_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    ranks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    npz_payload: dict[str, Any] = {
        "reference": ref,
        "pair": np.asarray(pair),
    }

    for method, samples in samples_by_method.items():
        n = min(len(samples), len(ref), args.n_posterior_samples)
        samples = samples[:n]
        reference = ref[:n]
        u_model, u_ref = separate_pair_ranks(samples, reference, pair)
        ranks[method] = (u_model, u_ref)
        npz_payload[f"{method}_samples"] = samples
        npz_payload[f"{method}_u_model"] = u_model
        npz_payload[f"{method}_u_ref"] = u_ref
        npz_payload[f"{method}_hist_model"] = hist2d(u_model, args.hist_bins)
        npz_payload[f"{method}_hist_ref"] = hist2d(u_ref, args.hist_bins)

        row = method_summary(
            method,
            args.obs,
            samples,
            reference,
            pair,
            hist_bins=args.hist_bins,
            angle_bins=args.angle_bins,
            c2st_folds=args.c2st_folds,
            c2st_max_epochs=args.c2st_max_epochs,
            seed=args.seed + args.obs,
        )
        summary_rows.append(row)
        conditional_rows.extend(
            conditional_slice_rows(
                method,
                args.obs,
                u_model,
                u_ref,
                n_bins=args.conditional_bins,
            )
        )
        tail_rows.extend(
            tail_quadrant_rows(
                method,
                args.obs,
                u_model,
                u_ref,
                thresholds=tail_thresholds,
            )
        )

    conditional_summary = aggregate_conditional(conditional_rows)
    tail_summary = aggregate_tail(tail_rows)
    summary_by_method = {
        row["method"]: {k: v for k, v in row.items() if k != "method"}
        for row in summary_rows
    }
    for row in conditional_summary:
        summary_by_method[row["method"]].update(
            {k: v for k, v in row.items() if k != "method"}
        )
    for row in tail_summary:
        summary_by_method[row["method"]].update(
            {k: v for k, v in row.items() if k != "method"}
        )
    combined_summary = [
        {"method": method, **vals} for method, vals in sorted(summary_by_method.items())
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method_tag = "-".join(methods)
    stem = (
        f"{args.task}_obs{args.obs}_pair{pair[0]}{pair[1]}_"
        f"n{args.n_train}_ps{args.n_posterior_samples}_{method_tag}_s{args.seed}"
    )
    summary_csv = out_dir / f"{stem}_summary.csv"
    conditional_csv = out_dir / f"{stem}_conditional.csv"
    tail_csv = out_dir / f"{stem}_tail.csv"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    fig_path = out_dir / f"{stem}.png"

    write_csv(summary_csv, combined_summary)
    write_csv(conditional_csv, conditional_rows)
    write_csv(tail_csv, tail_rows)
    np.savez(str(npz_path), **npz_payload)
    plot_diagnostic(fig_path, list(samples_by_method), ranks, conditional_rows, args.hist_bins)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    **vars(args),
                    "pair": pair,
                    "methods_expanded": list(samples_by_method),
                },
                "summary": combined_summary,
            },
            f,
            indent=2,
            default=str,
        )

    print("\nFocused Pairwise Copula Summary")
    print("-" * 112)
    print(
        f"{'method':<16} {'cop_c2st':>9} {'hist_tv':>8} {'cond_w':>8} "
        f"{'cond_ks':>8} {'tail_max':>9} {'radius_ks':>9} {'angle_tv':>8}"
    )
    for row in combined_summary:
        print(
            f"{row['method']:<16} "
            f"{row['pair_copula_c2st']:>9.4f} "
            f"{row['hist_tv']:>8.4f} "
            f"{row['mean_conditional_wasserstein']:>8.4f} "
            f"{row['mean_conditional_ks']:>8.4f} "
            f"{row['max_abs_tail_mass_diff']:>9.4f} "
            f"{row['radius_ks']:>9.4f} "
            f"{row['angle_tv']:>8.4f}"
        )
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {conditional_csv}")
    print(f"Wrote {tail_csv}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
