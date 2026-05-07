"""Interaction-order copula diagnostics for SBI posterior samples.

This script probes the failure mode suggested by the joint-structure analysis:
PFN-NPE can match marginal distributions and ordinary pairwise correlations,
while rank-space C2ST remains poor. The diagnostic can either remove marginal
differences with separate empirical ranks, giving a pure copula test, or use
pooled empirical ranks to preserve marginal rank differences as a comparison.
It then asks at which coordinate-subset order the two-sample classifier can
still separate approximate posterior samples from SBIBM reference samples.

It also clusters the reference posterior in Gaussianized copula space and
compares approximate-vs-reference cluster occupancies. This is a targeted test
for missing modes, wrong mode weights, or global copula geometry that may not be
visible in covariance/correlation summaries.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import norm
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    sample_posterior,
    train_flow,
)
from pfn_testing.sbi.sbibm_utils import compute_c2st, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import Config, PCAReducer, TabPFNEmbedder  # noqa: E402


def rank_columns(samples: np.ndarray) -> np.ndarray:
    """Map columns to empirical ranks in (0, 1)."""
    samples = np.asarray(samples, dtype=np.float64)
    ranks = np.empty_like(samples, dtype=np.float64)
    denom = samples.shape[0] + 1.0
    for d in range(samples.shape[1]):
        order = np.argsort(samples[:, d], kind="mergesort")
        ranks[order, d] = (np.arange(samples.shape[0]) + 1.0) / denom
    return ranks


def pooled_rank_transform(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map two sample sets to shared empirical ranks per dimension.

    Pooled ranks preserve one-dimensional marginal differences in rank space.
    They are useful as a diagnostic, but are not a pure copula transform.
    """
    n = min(len(samples_a), len(samples_b))
    pooled = np.concatenate([samples_a[:n], samples_b[:n]], axis=0)
    ranks = rank_columns(pooled)
    return ranks[:n], ranks[n:]


def separate_rank_transform(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each sample set to its own empirical ranks per dimension.

    Separate ranks remove all one-dimensional marginal information and compare
    only empirical copulas. Order-1 C2ST should be near chance.
    """
    n = min(len(samples_a), len(samples_b))
    return rank_columns(samples_a[:n]), rank_columns(samples_b[:n])


def rank_transform(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    rank_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if rank_mode == "separate":
        return separate_rank_transform(samples_a, samples_b)
    if rank_mode == "pooled":
        return pooled_rank_transform(samples_a, samples_b)
    raise ValueError(f"Unknown rank_mode: {rank_mode!r}")


def gaussianize_pooled_ranks(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    rank_mode: str,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank transform followed by normal-score transform."""
    a_rank, b_rank = rank_transform(samples_a, samples_b, rank_mode)
    return norm.ppf(np.clip(a_rank, eps, 1.0 - eps)), norm.ppf(
        np.clip(b_rank, eps, 1.0 - eps)
    )


def subset_list(
    dim: int,
    order: int,
    max_subsets: int | None,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Return all or a deterministic random subset of coordinate combinations."""
    combos = list(itertools.combinations(range(dim), order))
    if max_subsets is not None and len(combos) > max_subsets:
        idx = rng.choice(len(combos), size=max_subsets, replace=False)
        combos = [combos[i] for i in sorted(idx)]
    return combos


def interaction_order_c2st(
    samples: np.ndarray,
    reference: np.ndarray,
    rank_mode: str,
    max_order: int,
    max_subsets: int | None,
    c2st_folds: int,
    c2st_max_epochs: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run rank-space C2ST on coordinate subsets of increasing order."""
    n = min(len(samples), len(reference))
    samples_rank, ref_rank = rank_transform(samples[:n], reference[:n], rank_mode)
    dim = samples_rank.shape[1]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for order in range(1, min(max_order, dim) + 1):
        for subset in subset_list(dim, order, max_subsets, rng):
            a = samples_rank[:, subset]
            b = ref_rank[:, subset]
            score = compute_c2st(
                a,
                b,
                n_folds=c2st_folds,
                max_epochs=c2st_max_epochs,
                seed=seed,
            )
            rows.append(
                {
                    "order": order,
                    "rank_mode": rank_mode,
                    "subset": " ".join(str(i) for i in subset),
                    "c2st": float(score),
                }
            )

    if max_order < dim:
        full_subset = tuple(range(dim))
        score = compute_c2st(
            samples_rank,
            ref_rank,
            n_folds=c2st_folds,
            max_epochs=c2st_max_epochs,
            seed=seed,
        )
        rows.append(
            {
                "order": dim,
                "rank_mode": rank_mode,
                "subset": " ".join(str(i) for i in full_subset),
                "c2st": float(score),
            }
        )

    return rows


def cluster_occupancy(
    samples: np.ndarray,
    reference: np.ndarray,
    rank_mode: str,
    n_clusters: int,
    seed: int,
    missing_threshold: float,
) -> dict[str, Any]:
    """Compare cluster weights in Gaussianized rank/copula space."""
    n = min(len(samples), len(reference))
    z_model, z_ref = gaussianize_pooled_ranks(samples[:n], reference[:n], rank_mode)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    ref_labels = kmeans.fit_predict(z_ref)
    model_labels = kmeans.predict(z_model)

    ref_counts = np.bincount(ref_labels, minlength=n_clusters).astype(np.float64)
    model_counts = np.bincount(model_labels, minlength=n_clusters).astype(np.float64)
    ref_w = ref_counts / ref_counts.sum()
    model_w = model_counts / model_counts.sum()
    smooth = 0.5 / n_clusters
    ref_s = (ref_counts + smooth) / (ref_counts.sum() + smooth * n_clusters)
    model_s = (model_counts + smooth) / (model_counts.sum() + smooth * n_clusters)

    tv = 0.5 * np.abs(model_w - ref_w).sum()
    missing = (model_w < missing_threshold) & (ref_w >= missing_threshold)
    over = np.maximum(model_w - ref_w, 0.0).sum()
    under = np.maximum(ref_w - model_w, 0.0).sum()

    return {
        "n_clusters": n_clusters,
        "rank_mode": rank_mode,
        "occupancy_tv": float(tv),
        "max_abs_weight_error": float(np.abs(model_w - ref_w).max()),
        "missing_cluster_count": int(missing.sum()),
        "missing_ref_mass": float(ref_w[missing].sum()),
        "over_mass": float(over),
        "under_mass": float(under),
        "kl_ref_to_model": float(np.sum(ref_s * np.log(ref_s / model_s))),
        "ref_weights": " ".join(f"{x:.6f}" for x in ref_w),
        "model_weights": " ".join(f"{x:.6f}" for x in model_w),
    }


def summarize_order_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r["method"], r["rank_mode"], int(r["order"])) for r in rows})
    for method, rank_mode, order in keys:
        vals = np.asarray(
            [
                float(r["c2st"])
                for r in rows
                if r["method"] == method
                and r["rank_mode"] == rank_mode
                and int(r["order"]) == order
            ],
            dtype=np.float64,
        )
        out.append(
            {
                "method": method,
                "rank_mode": rank_mode,
                "order": order,
                "n_tests": int(len(vals)),
                "c2st_mean": float(vals.mean()),
                "c2st_std": float(vals.std()),
                "c2st_min": float(vals.min()),
                "c2st_q90": float(np.quantile(vals, 0.9)),
                "c2st_max": float(vals.max()),
            }
        )
    return out


def summarize_occupancy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r["method"], r["rank_mode"], int(r["n_clusters"])) for r in rows})
    metrics = [
        "occupancy_tv",
        "max_abs_weight_error",
        "missing_cluster_count",
        "missing_ref_mass",
        "kl_ref_to_model",
    ]
    for method, rank_mode, n_clusters in keys:
        subset = [
            r
            for r in rows
            if r["method"] == method
            and r["rank_mode"] == rank_mode
            and int(r["n_clusters"]) == n_clusters
        ]
        row: dict[str, Any] = {
            "method": method,
            "rank_mode": rank_mode,
            "n_clusters": n_clusters,
            "n_obs": len(subset),
        }
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in subset], dtype=np.float64)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std())
        out.append(row)
    return out


def train_flow_for_context(
    method: str,
    context_train: np.ndarray,
    context_val: np.ndarray,
    theta_train: np.ndarray,
    theta_val: np.ndarray,
    cfg: Config,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    dim_theta = theta_train.shape[1]
    defaults = get_flow_defaults(dim_theta)
    n_transforms = cfg.n_transforms or defaults["n_transforms"]
    hidden_features = cfg.hidden_features or defaults["hidden_features"]
    print(
        f"\nTraining {method} flow: context_dim={context_train.shape[1]} "
        f"transforms={n_transforms} hidden={hidden_features}"
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    flow = build_flow(
        dim_theta=dim_theta,
        dim_context=context_train.shape[1],
        n_transforms=n_transforms,
        hidden_features=hidden_features,
        n_bins=cfg.n_bins,
        flow_type=cfg.flow_type,
    )
    history = train_flow(flow, theta_train, context_train, theta_val, context_val, cfg)
    return flow, history


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", default="tabpfn", help="Comma list: tabpfn,raw")
    ap.add_argument("--context-size", type=int, default=1_000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--label-strategy", default="per_dim", choices=["per_dim", "per_dim_mean"])
    ap.add_argument("--model-type", default="regressor", choices=["regressor", "classifier"])
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"])
    ap.add_argument("--flow-type", default="nsf", choices=["nsf", "naf"])
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-posterior-samples", type=int, default=1_000)
    ap.add_argument("--n-reference-observations", type=int, default=10)
    ap.add_argument("--max-order", type=int, default=3)
    ap.add_argument("--max-subsets", type=int, default=None)
    ap.add_argument(
        "--rank-mode",
        default="separate",
        choices=["separate", "pooled", "both"],
        help=(
            "separate removes one-dimensional marginals and is the pure "
            "copula test; pooled preserves marginal rank differences."
        ),
    )
    ap.add_argument("--c2st-folds", type=int, default=3)
    ap.add_argument("--c2st-max-epochs", type=int, default=100)
    ap.add_argument("--clusters", default="4,6,8")
    ap.add_argument("--missing-threshold", type=float, default=0.01)
    ap.add_argument("--out-dir", default="pfn_testing/sbi/outputs/interaction_order_copula")
    args = ap.parse_args()

    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(requested_methods) - {"tabpfn", "raw"})
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}; expected tabpfn and/or raw")

    print(
        f"Simulating {args.task}: n_train={args.n_train}, n_val={args.n_val}, "
        f"seed={args.seed}"
    )
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    task = data["task"]
    dim_theta = data["dim_theta"]
    obs_x = np.stack(
        [
            task.get_observation(num_observation=i).numpy().squeeze(0)
            for i in range(1, args.n_reference_observations + 1)
        ],
        axis=0,
    ).astype(np.float32)

    contexts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if "raw" in requested_methods:
        contexts["raw_x"] = (data["xs_train"], data["xs_val"], obs_x)

    if "tabpfn" in requested_methods:
        print("\nExtracting TabPFN summaries")
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
        emb_obs = embedder.transform(obs_x)
        print(
            f"  full embedding shapes: train={emb_train.shape} "
            f"val={emb_val.shape} obs={emb_obs.shape}"
        )
        pca = PCAReducer(args.embed_dim)
        contexts[f"tabpfn_pca{args.embed_dim}"] = (
            pca.fit_transform(emb_train),
            pca.transform(emb_val),
            pca.transform(emb_obs),
        )

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
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        n_posterior_samples=args.n_posterior_samples,
        n_reference_observations=args.n_reference_observations,
        skip_raw=True,
    )

    order_rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    sample_shapes: dict[str, Any] = {}
    clusters = parse_int_list(args.clusters)
    rank_modes = ["separate", "pooled"] if args.rank_mode == "both" else [args.rank_mode]

    refs = {
        obs_id: task.get_reference_posterior_samples(num_observation=obs_id)
        .numpy()[: args.n_posterior_samples]
        for obs_id in range(1, args.n_reference_observations + 1)
    }

    for method, (context_train, context_val, context_obs) in contexts.items():
        flow, history = train_flow_for_context(
            method,
            context_train,
            context_val,
            data["thetas_train"],
            data["thetas_val"],
            cfg,
        )
        sample_shapes[method] = {}
        for obs_id in range(1, args.n_reference_observations + 1):
            print(f"  evaluating {method}, obs {obs_id}")
            samples = sample_posterior(
                flow,
                context_obs[obs_id - 1],
                history["theta_mean"],
                history["theta_std"],
                args.n_posterior_samples,
            )
            ref = refs[obs_id]
            sample_shapes[method][obs_id] = samples.shape

            for rank_mode in rank_modes:
                rows = interaction_order_c2st(
                    samples,
                    ref,
                    rank_mode=rank_mode,
                    max_order=args.max_order,
                    max_subsets=args.max_subsets,
                    c2st_folds=args.c2st_folds,
                    c2st_max_epochs=args.c2st_max_epochs,
                    seed=args.seed + obs_id,
                )
                for row in rows:
                    row.update({"method": method, "obs": obs_id})
                    order_rows.append(row)

                for k in clusters:
                    if k >= min(len(samples), len(ref)):
                        continue
                    row = cluster_occupancy(
                        samples,
                        ref,
                        rank_mode=rank_mode,
                        n_clusters=k,
                        seed=args.seed + 10_000 * obs_id + k,
                        missing_threshold=args.missing_threshold,
                    )
                    row.update({"method": method, "obs": obs_id})
                    occupancy_rows.append(row)

    order_summary = summarize_order_rows(order_rows)
    occupancy_summary = summarize_occupancy_rows(occupancy_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method_tag = "-".join(requested_methods)
    stem = (
        f"{args.task}_n{args.n_train}_s{args.seed}_{method_tag}_"
        f"pca{args.embed_dim}_ps{args.n_posterior_samples}_{args.rank_mode}"
    )
    order_csv = out_dir / f"{stem}_order.csv"
    order_summary_csv = out_dir / f"{stem}_order_summary.csv"
    occupancy_csv = out_dir / f"{stem}_occupancy.csv"
    occupancy_summary_csv = out_dir / f"{stem}_occupancy_summary.csv"
    json_path = out_dir / f"{stem}.json"

    write_csv(order_csv, order_rows)
    write_csv(order_summary_csv, order_summary)
    write_csv(occupancy_csv, occupancy_rows)
    write_csv(occupancy_summary_csv, occupancy_summary)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    **asdict(cfg),
                    "model_version": args.model_version,
                    "methods": requested_methods,
                    "max_order": args.max_order,
                    "max_subsets": args.max_subsets,
                    "rank_mode": args.rank_mode,
                    "clusters": clusters,
                    "c2st_folds": args.c2st_folds,
                    "c2st_max_epochs": args.c2st_max_epochs,
                },
                "dim_theta": dim_theta,
                "sample_shapes": sample_shapes,
                "order_summary": order_summary,
                "occupancy_summary": occupancy_summary,
            },
            f,
            indent=2,
            default=str,
        )

    print("\nInteraction-Order Rank C2ST")
    print("-" * 72)
    print(
        f"{'method':<16} {'rank':<8} {'order':>5} {'mean':>8} "
        f"{'q90':>8} {'max':>8} {'n':>5}"
    )
    for row in order_summary:
        print(
            f"{row['method']:<16} {row['rank_mode']:<8} {row['order']:>5} "
            f"{row['c2st_mean']:>8.4f} {row['c2st_q90']:>8.4f} "
            f"{row['c2st_max']:>8.4f} {row['n_tests']:>5}"
        )

    print("\nMode Occupancy")
    print("-" * 72)
    print(f"{'method':<16} {'rank':<8} {'K':>4} {'TV':>8} {'miss_mass':>10} {'KL':>8}")
    for row in occupancy_summary:
        print(
            f"{row['method']:<16} {row['rank_mode']:<8} {row['n_clusters']:>4} "
            f"{row['occupancy_tv_mean']:>8.4f} "
            f"{row['missing_ref_mass_mean']:>10.4f} "
            f"{row['kl_ref_to_model_mean']:>8.4f}"
        )

    print(f"\nWrote {order_csv}")
    print(f"Wrote {order_summary_csv}")
    print(f"Wrote {occupancy_csv}")
    print(f"Wrote {occupancy_summary_csv}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
