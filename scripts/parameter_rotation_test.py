"""Parameter-rotation sensitivity test for PFN-NPE.

The PFN-NPE default label strategy runs TabPFN once per parameter coordinate
and concatenates the resulting embeddings. This construction may be sensitive
to the arbitrary coordinate system used for theta. This script tests that
sensitivity by training the same PFN-NPE pipeline on rotated parameters

    z = theta @ Q

where Q is an orthogonal matrix. Posterior samples are drawn in z-space and
rotated back to theta-space before comparison with the original reference
posterior. The simulator, observations, and reference posterior are unchanged.

If PFN-NPE were approximately invariant to parameterization, identity and
random rotations would have similar joint, marginal, and rank C2ST. Large
changes indicate that the coordinate-wise label construction, the projection,
or the downstream flow depends on the chosen parameter basis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    sample_posterior,
    train_flow,
)
from pfn_testing.sbi.sbibm_utils import compute_c2st, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import (  # noqa: E402
    Config,
    LinearProjectionReducer,
    PCAReducer,
    TabPFNEmbedder,
)


def make_rotation(dim: int, seed: int) -> np.ndarray:
    """Generate a deterministic proper orthogonal rotation matrix."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def rank_transform_pair(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map two sample sets to pooled empirical ranks per dimension."""
    n = min(len(samples_a), len(samples_b))
    a = np.asarray(samples_a[:n], dtype=np.float64)
    b = np.asarray(samples_b[:n], dtype=np.float64)
    pooled = np.concatenate([a, b], axis=0)
    ranks = np.empty_like(pooled, dtype=np.float64)
    denom = pooled.shape[0] + 1.0
    for d in range(pooled.shape[1]):
        order = np.argsort(pooled[:, d], kind="mergesort")
        ranks[order, d] = (np.arange(pooled.shape[0]) + 1.0) / denom
    return ranks[:n], ranks[n:]


def evaluate_decomposition(
    task: object,
    sample_fn,
    n_obs: int,
    n_posterior_samples: int,
    c2st_folds: int = 5,
    c2st_max_epochs: int = 500,
    c2st_seed: int = 1,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Compute joint, marginal, and rank-space C2ST per observation."""
    joint: list[float] = []
    marginal: list[float] = []
    rank: list[float] = []
    marginal_per_dim: list[list[float]] = []

    for obs_id in range(1, n_obs + 1):
        x_obs = task.get_observation(num_observation=obs_id).numpy().squeeze(0)
        ref = task.get_reference_posterior_samples(num_observation=obs_id).numpy()
        posterior = sample_fn(x_obs)
        if len(posterior) > n_posterior_samples:
            posterior = posterior[:n_posterior_samples]
        ref = ref[: max(len(posterior), n_posterior_samples)]

        joint_score = compute_c2st(
            posterior,
            ref,
            n_folds=c2st_folds,
            max_epochs=c2st_max_epochs,
            seed=c2st_seed,
        )
        per_dim = [
            compute_c2st(
                posterior[:, d : d + 1],
                ref[:, d : d + 1],
                n_folds=c2st_folds,
                max_epochs=c2st_max_epochs,
                seed=c2st_seed,
            )
            for d in range(posterior.shape[1])
        ]
        posterior_rank, ref_rank = rank_transform_pair(posterior, ref)
        rank_score = compute_c2st(
            posterior_rank,
            ref_rank,
            n_folds=c2st_folds,
            max_epochs=c2st_max_epochs,
            seed=c2st_seed,
        )

        joint.append(joint_score)
        marginal.append(float(np.mean(per_dim)))
        marginal_per_dim.append(per_dim)
        rank.append(rank_score)

        if verbose:
            print(
                f"    obs {obs_id:02d}: joint={joint_score:.4f} "
                f"marginal={np.mean(per_dim):.4f} rank={rank_score:.4f}"
            )

    return {
        "joint": np.asarray(joint, dtype=np.float64),
        "marginal": np.asarray(marginal, dtype=np.float64),
        "rank": np.asarray(rank, dtype=np.float64),
        "marginal_per_dim": np.asarray(marginal_per_dim, dtype=np.float64),
    }


def reduce_embeddings(
    emb_train: np.ndarray,
    emb_val: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, object | None, int]:
    """Apply the same fixed embedding reduction used by PFN-NPE."""
    emb_dim = emb_train.shape[1]
    reducer = None

    if cfg.embed_dim is None:
        return emb_train, emb_val, reducer, emb_dim

    if cfg.reduction == "pca":
        reducer = PCAReducer(cfg.embed_dim)
    elif cfg.reduction == "linear":
        reducer = LinearProjectionReducer(cfg.embed_dim)
    elif cfg.reduction == "truncate":
        return emb_train[:, : cfg.embed_dim], emb_val[:, : cfg.embed_dim], None, cfg.embed_dim
    else:
        raise ValueError(
            "parameter_rotation_test supports reduction in "
            "{pca, linear, truncate} or embed_dim=None; "
            f"got {cfg.reduction!r}"
        )

    emb_train = reducer.fit_transform(emb_train)
    emb_val = reducer.transform(emb_val)
    return emb_train, emb_val, reducer, cfg.embed_dim


def set_training_seed(seed: int) -> None:
    """Reset stochastic training state so rotations are compared fairly."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_rotation(
    data: dict,
    q: np.ndarray,
    rotation_name: str,
    cfg: Config,
    flow_targets: list[str],
) -> list[dict]:
    """Train PFN-NPE with rotated labels and evaluate in theta-space."""
    task = data["task"]
    dim_theta = data["dim_theta"]
    z_train = data["thetas_train"] @ q
    z_val = data["thetas_val"] @ q

    print("\n" + "=" * 72)
    print(f"Rotation: {rotation_name}")
    print("=" * 72)
    print("  extracting TabPFN embeddings using rotated theta labels")
    embedder = TabPFNEmbedder(
        context_size=cfg.context_size,
        seed=cfg.seed,
        label_strategy=cfg.label_strategy,
        layer=cfg.layer,
        model_type=cfg.model_type,
        device=cfg.device,
        mrl_dim=cfg.mrl_dim,
        model_version=getattr(cfg, "model_version", "v2"),
    )
    embedder.fit(data["xs_train"], thetas=z_train)
    emb_train = embedder.transform(data["xs_train"])
    emb_val = embedder.transform(data["xs_val"])
    full_emb_dim = emb_train.shape[1]
    print(f"  full embeddings: train={emb_train.shape} val={emb_val.shape}")

    emb_train, emb_val, reducer, flow_context_dim = reduce_embeddings(
        emb_train, emb_val, cfg,
    )
    print(f"  flow context dim: {flow_context_dim}")

    defaults = get_flow_defaults(dim_theta)
    n_transforms = cfg.n_transforms or defaults["n_transforms"]
    hidden_features = cfg.hidden_features or defaults["hidden_features"]
    results = []

    for flow_target in flow_targets:
        if flow_target == "rotated":
            theta_train_flow = z_train
            theta_val_flow = z_val
        elif flow_target == "original":
            theta_train_flow = data["thetas_train"]
            theta_val_flow = data["thetas_val"]
        else:
            raise ValueError(f"Unknown flow_target: {flow_target!r}")

        print("\n" + "-" * 72)
        print(f"  flow target: {flow_target}")
        print("-" * 72)

        set_training_seed(cfg.seed)
        flow = build_flow(
            dim_theta=dim_theta,
            dim_context=flow_context_dim,
            n_transforms=n_transforms,
            hidden_features=hidden_features,
            n_bins=cfg.n_bins,
            flow_type=cfg.flow_type,
        )
        print(
            f"  training {cfg.flow_type.upper()} on {flow_target} theta target "
            f"(transforms={n_transforms}, hidden={hidden_features})"
        )
        history = train_flow(
            flow,
            theta_train_flow,
            emb_train,
            theta_val_flow,
            emb_val,
            cfg,
        )

        def sample_fn(x_obs: np.ndarray) -> np.ndarray:
            obs_emb = embedder.transform(x_obs.reshape(1, -1))
            if reducer is not None:
                obs_emb = reducer.transform(obs_emb)
            elif cfg.reduction == "truncate" and cfg.embed_dim is not None:
                obs_emb = obs_emb[:, : cfg.embed_dim]
            samples = sample_posterior(
                flow,
                obs_emb[0],
                history["theta_mean"],
                history["theta_std"],
                cfg.n_posterior_samples,
            )
            if flow_target == "rotated":
                return samples @ q.T
            return samples

        print("  evaluating posterior samples in theta-space")
        metrics = evaluate_decomposition(
            task,
            sample_fn,
            cfg.n_reference_observations,
            cfg.n_posterior_samples,
            c2st_folds=getattr(cfg, "c2st_folds", 5),
            c2st_max_epochs=getattr(cfg, "c2st_max_epochs", 500),
            c2st_seed=getattr(cfg, "c2st_seed", 1),
        )
        results.append(
            {
                "rotation": rotation_name,
                "flow_target": flow_target,
                "q": q,
                "full_emb_dim": full_emb_dim,
                "flow_context_dim": flow_context_dim,
                "best_epoch": history["best_epoch"],
                **metrics,
            }
        )

    return results


def summarize(results: list[dict]) -> list[dict]:
    rows = []

    for r in results:
        identity = next(
            (
                base
                for base in results
                if base["rotation"] == "identity"
                and base["flow_target"] == r["flow_target"]
            ),
            None,
        )
        base_joint = None if identity is None else float(identity["joint"].mean())
        base_marginal = None if identity is None else float(identity["marginal"].mean())
        base_rank = None if identity is None else float(identity["rank"].mean())
        row = {
            "rotation": r["rotation"],
            "flow_target": r["flow_target"],
            "joint_mean": float(r["joint"].mean()),
            "joint_std": float(r["joint"].std()),
            "marginal_mean": float(r["marginal"].mean()),
            "marginal_std": float(r["marginal"].std()),
            "rank_mean": float(r["rank"].mean()),
            "rank_std": float(r["rank"].std()),
            "joint_minus_marginal": float(
                r["joint"].mean() - r["marginal"].mean()
            ),
            "best_epoch": int(r["best_epoch"]),
        }
        if base_joint is not None:
            row["delta_joint_vs_identity"] = row["joint_mean"] - base_joint
            row["delta_marginal_vs_identity"] = row["marginal_mean"] - base_marginal
            row["delta_rank_vs_identity"] = row["rank_mean"] - base_rank
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--context-size", type=int, default=1_000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--reduction", default="pca", choices=["pca", "linear", "truncate"])
    ap.add_argument("--label-strategy", default="per_dim", choices=["per_dim", "per_dim_mean"])
    ap.add_argument("--model-type", default="regressor", choices=["regressor", "classifier"])
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--mrl-dim", type=int, default=None)
    ap.add_argument("--flow-type", default="nsf", choices=["nsf", "naf"])
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-posterior-samples", type=int, default=1_000)
    ap.add_argument("--n-reference-observations", type=int, default=10)
    ap.add_argument("--n-rotations", type=int, default=3)
    ap.add_argument("--rotation-seed", type=int, default=20260504)
    ap.add_argument(
        "--flow-target",
        default="original",
        choices=["original", "rotated", "both"],
        help=(
            "original: train the flow on original theta while rotating only "
            "TabPFN labels; rotated: train end-to-end in rotated coordinates; "
            "both: run both modes"
        ),
    )
    ap.add_argument("--c2st-folds", type=int, default=5)
    ap.add_argument("--c2st-max-epochs", type=int, default=500)
    ap.add_argument("--c2st-seed", type=int, default=1)
    ap.add_argument(
        "--out-dir",
        default="pfn_testing/sbi/outputs/parameter_rotation",
    )
    args = ap.parse_args()

    cfg = Config(
        task_name=args.task,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        device=args.device,
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        reduction=args.reduction,
        label_strategy=args.label_strategy,
        layer=args.layer,
        model_type=args.model_type,
        mrl_dim=args.mrl_dim,
        flow_type=args.flow_type,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        n_posterior_samples=args.n_posterior_samples,
        n_reference_observations=args.n_reference_observations,
        skip_raw=True,
    )
    # main has model-version support on the embedder but Config does not store it.
    setattr(cfg, "model_version", args.model_version)
    setattr(cfg, "c2st_folds", args.c2st_folds)
    setattr(cfg, "c2st_max_epochs", args.c2st_max_epochs)
    setattr(cfg, "c2st_seed", args.c2st_seed)

    print(f"Simulating shared data: task={args.task} n_train={args.n_train}")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = data["dim_theta"]
    if dim_theta < 2:
        raise SystemExit("Parameter rotation test requires dim_theta >= 2")

    rotations: list[tuple[str, np.ndarray]] = [
        ("identity", np.eye(dim_theta, dtype=np.float32))
    ]
    for k in range(args.n_rotations):
        rotations.append(
            (
                f"rotation_{k}",
                make_rotation(dim_theta, args.rotation_seed + 1009 * k),
            )
        )

    flow_targets = (
        ["original", "rotated"] if args.flow_target == "both" else [args.flow_target]
    )
    results = []
    for name, q in rotations:
        results.extend(train_one_rotation(data, q, name, cfg, flow_targets))
    rows = summarize(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.task}_n{args.n_train}_s{args.seed}_"
        f"{args.label_strategy}_{args.reduction}{args.embed_dim}_"
        f"r{args.n_rotations}_{args.flow_target}"
    )
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    save_payload = {
        "task": args.task,
        "config": json.dumps(
            {
                **asdict(cfg),
                "model_version": args.model_version,
                "flow_target": args.flow_target,
                "c2st_folds": args.c2st_folds,
                "c2st_max_epochs": args.c2st_max_epochs,
                "c2st_seed": args.c2st_seed,
            },
            default=str,
        ),
    }
    for r in results:
        prefix = f"{r['flow_target']}_{r['rotation']}"
        save_payload[f"{prefix}_q"] = r["q"]
        save_payload[f"{prefix}_joint"] = r["joint"]
        save_payload[f"{prefix}_marginal"] = r["marginal"]
        save_payload[f"{prefix}_rank"] = r["rank"]
        save_payload[f"{prefix}_marginal_per_dim"] = r["marginal_per_dim"]
    np.savez(str(npz_path), **save_payload)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    **asdict(cfg),
                    "model_version": args.model_version,
                    "flow_target": args.flow_target,
                    "c2st_folds": args.c2st_folds,
                    "c2st_max_epochs": args.c2st_max_epochs,
                    "c2st_seed": args.c2st_seed,
                },
                "summary": rows,
            },
            f,
            indent=2,
            default=str,
        )

    header = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    print("\nSummary")
    print("-" * 72)
    print(
        f"{'target':<9} {'rotation':<12} {'joint':>8} {'marg':>8} "
        f"{'rank':>8} {'j-m':>8} {'d_joint':>9}"
    )
    for row in rows:
        print(
            f"{row['flow_target']:<9} {row['rotation']:<12} "
            f"{row['joint_mean']:>8.4f} "
            f"{row['marginal_mean']:>8.4f} {row['rank_mean']:>8.4f} "
            f"{row['joint_minus_marginal']:>8.4f} "
            f"{row.get('delta_joint_vs_identity', 0.0):>9.4f}"
        )
    print(f"\nWrote {npz_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
