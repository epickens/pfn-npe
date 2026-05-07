"""PFN-weighted fixed-table NLE.

This experiment tests the budget-matched localization idea: use TabPFN/PFN-NPE
only to select or reweight rows from an existing prior simulation table, then fit
an observation-specific NLE without drawing any additional simulator samples.

The simulator budget is therefore the same global table used by the amortized
sbi baselines. The tradeoff is extra per-observation computation, not extra
simulations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfn_proposal_nle_refinement import (  # noqa: E402
    evaluate_method,
    get_uniform_prior_bounds,
    proposal_box,
    train_local_nle,
)
from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    sample_posterior,
    train_flow,
)
from pfn_testing.sbi.sbibm_utils import simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import Config, PCAReducer, TabPFNEmbedder  # noqa: E402


SBI_BASELINES = {
    "sbi-NPE": ("sbi_npe", "sbi_npe_joint_c2st", "sbi_npe_marginal_c2st", "sbi_npe_rank_c2st"),
    "sbi-NLE": ("sbi_nle", "sbi_nle_joint_c2st", "sbi_nle_marginal_c2st", "sbi_nle_rank_c2st"),
    "sbi-NRE": ("sbi_nre", "sbi_nre_joint_c2st", "sbi_nre_marginal_c2st", "sbi_nre_rank_c2st"),
    "sbi-FMPE": ("sbi_fmpe", "sbi_fmpe_joint_c2st", "sbi_fmpe_marginal_c2st", "sbi_fmpe_rank_c2st"),
}
CONTEXT_BASELINES = {
    "BayesFlow": ("bayesflow", "bayesflow_joint_c2st", "bayesflow_marginal_c2st", "bayesflow_rank_c2st"),
    "PFN-NPE PCA64": (
        "n10000_per_dim_regressor_pca_64",
        "tabpfn_joint_c2st",
        "tabpfn_marginal_c2st",
        "tabpfn_rank_c2st",
    ),
    "Raw-x NSF": (
        "n10000_per_dim_regressor_pca_64",
        "raw_joint_c2st",
        "raw_marginal_c2st",
        "raw_rank_c2st",
    ),
}


@dataclass
class PFNProposalModel:
    embedder: TabPFNEmbedder
    reducer: PCAReducer
    flow: torch.nn.Module
    theta_mean: np.ndarray
    theta_std: np.ndarray


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def fit_pfn_proposal_model(data: dict[str, Any], args: argparse.Namespace) -> PFNProposalModel:
    print("\n[1/3] Fitting amortized PFN proposal on fixed table")
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

    reducer = PCAReducer(args.embed_dim)
    ctx_train = reducer.fit_transform(emb_train)
    ctx_val = reducer.transform(emb_val)

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
        n_posterior_samples=args.n_proposal_samples,
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
    return PFNProposalModel(
        embedder=embedder,
        reducer=reducer,
        flow=flow,
        theta_mean=history["theta_mean"],
        theta_std=history["theta_std"],
    )


def sample_pfn_proposal(
    model: PFNProposalModel,
    obs_x: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    emb_obs = model.embedder.transform(obs_x.reshape(1, -1))
    ctx_obs = model.reducer.transform(emb_obs)[0]
    samples = sample_posterior(
        model.flow,
        ctx_obs,
        model.theta_mean,
        model.theta_std,
        n_samples,
    )
    return samples.astype(np.float32)


def proposal_log_weights(
    theta_train: np.ndarray,
    proposal_samples: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    ridge_frac: float,
) -> np.ndarray:
    in_prior = np.all(
        (proposal_samples >= prior_low) & (proposal_samples <= prior_high),
        axis=1,
    )
    samples = proposal_samples[in_prior]
    if len(samples) < max(100, 10 * theta_train.shape[1]):
        samples = np.clip(proposal_samples, prior_low, prior_high)

    center = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]])
    width = prior_high - prior_low
    cov = cov + np.diag((ridge_frac * width) ** 2)
    inv_cov = np.linalg.pinv(cov)
    diff = theta_train - center
    maha = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
    return -0.5 * maha


def normalize_log_weights(logw: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logw / max(temperature, 1e-8)
    scaled = scaled - np.max(scaled)
    w = np.exp(scaled)
    return w / np.sum(w)


def ess(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def local_box_from_theta(
    theta: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    pad_frac: float,
    min_width_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_lo = np.quantile(theta, 0.02, axis=0)
    q_hi = np.quantile(theta, 0.98, axis=0)
    width = q_hi - q_lo
    lo = q_lo - pad_frac * width
    hi = q_hi + pad_frac * width

    # The local prior must contain every selected simulation row. The quantile
    # box stabilizes against distant selected tails, but the final support
    # still has to include all rows passed to sbi-NLE.
    lo = np.minimum(lo, theta.min(axis=0))
    hi = np.maximum(hi, theta.max(axis=0))

    min_width = min_width_frac * (prior_high - prior_low)
    center = 0.5 * (lo + hi)
    narrow = (hi - lo) < min_width
    lo[narrow] = center[narrow] - 0.5 * min_width[narrow]
    hi[narrow] = center[narrow] + 0.5 * min_width[narrow]

    return (
        np.maximum(lo, prior_low).astype(np.float32),
        np.minimum(hi, prior_high).astype(np.float32),
    )


def select_fixed_table_rows(
    theta_train: np.ndarray,
    x_train: np.ndarray,
    logw: np.ndarray,
    n_select: int,
    strategy: str,
    temperature: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n_select = min(n_select, len(theta_train))
    rng = np.random.default_rng(seed)
    probs = normalize_log_weights(logw, temperature)

    if strategy == "topk":
        idx = np.argpartition(-logw, n_select - 1)[:n_select]
        idx = idx[np.argsort(-logw[idx])]
    elif strategy == "resample":
        idx = rng.choice(len(theta_train), size=n_select, replace=True, p=probs)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy!r}")

    return (
        theta_train[idx].astype(np.float32),
        x_train[idx].astype(np.float32),
        {
            "selected_count": float(len(idx)),
            "selected_unique_count": float(len(np.unique(idx))),
            "weight_ess": ess(probs),
            "max_weight": float(np.max(probs)),
            "mean_selected_logw": float(np.mean(logw[idx])),
            "min_selected_logw": float(np.min(logw[idx])),
        },
    )


def train_fixed_table_nle(
    task: object,
    theta_selected: np.ndarray,
    x_selected: np.ndarray,
    box_low: np.ndarray,
    box_high: np.ndarray,
    obs_x: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    local_data = {
        "theta_train": theta_selected,
        "x_train": x_selected,
        "theta_val": theta_selected[:0],
        "x_val": x_selected[:0],
    }
    return train_local_nle(task, local_data, box_low, box_high, obs_x, args)


def baseline_npz_path(task: str, tag: str, seed: int, n_train: int, ps: int) -> Path:
    base = Path("pfn_testing/sbi/outputs") / task
    if tag.startswith("sbi_") or tag == "bayesflow":
        suffix = f"{tag}_n{n_train}_ps{ps}"
        if seed != 42:
            suffix += f"_s{seed}"
        path = base / suffix / "results" / "results.npz"
        if path.exists():
            return path
        return base / f"{tag}_n{n_train}" / "results" / "results.npz"

    suffix = f"{tag}_ps{ps}_s{seed}"
    path = base / suffix / "results" / "results.npz"
    if path.exists():
        return path
    return base / tag / "results" / "results.npz"


def load_baselines(args: argparse.Namespace, obs: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baselines = dict(SBI_BASELINES)
    if args.include_context_baselines:
        baselines.update(CONTEXT_BASELINES)
    obs_idx = np.asarray(obs) - 1
    for label, (tag, joint_key, marginal_key, rank_key) in baselines.items():
        path = baseline_npz_path(
            args.task,
            tag,
            args.seed,
            args.n_train,
            args.n_posterior_samples,
        )
        if not path.exists():
            print(f"[baseline] missing {label}: {path}")
            continue
        data = np.load(path, allow_pickle=True)
        rows.append(
            {
                "method": label,
                "training_simulations_total": args.n_train,
                "training_simulations_per_obs": args.n_train / len(obs),
                "joint_mean": float(np.mean(data[joint_key][obs_idx])),
                "marginal_mean": float(np.mean(data[marginal_key][obs_idx])),
                "rank_mean": float(np.mean(data[rank_key][obs_idx])),
                "joint_std_obs": float(np.std(data[joint_key][obs_idx], ddof=0)),
                "marginal_std_obs": float(np.std(data[marginal_key][obs_idx], ddof=0)),
                "rank_std_obs": float(np.std(data[rank_key][obs_idx], ddof=0)),
            }
        )
    return rows


def print_comparison(rows: list[dict[str, Any]]) -> None:
    print("\nPFN-weighted fixed-table comparison")
    print("-" * 92)
    print(
        f"{'method':<34} {'sims total':>11} {'sims/obs':>10} "
        f"{'joint':>8} {'marg':>8} {'rank':>8}"
    )
    for row in sorted(rows, key=lambda r: r["joint_mean"]):
        print(
            f"{row['method']:<34} "
            f"{row['training_simulations_total']:>11.0f} "
            f"{row['training_simulations_per_obs']:>10.0f} "
            f"{row['joint_mean']:>8.4f} "
            f"{row['marginal_mean']:>8.4f} "
            f"{row['rank_mean']:>8.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--obs", type=int, nargs="+", default=[1])
    ap.add_argument("--pair", default="0,1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--context-size", type=int, default=1_000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--label-strategy", default="per_dim", choices=["per_dim", "per_dim_mean"])
    ap.add_argument("--model-type", default="regressor", choices=["regressor", "classifier"])
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"])
    ap.add_argument("--flow-type", default="nsf", choices=["nsf", "naf"])
    ap.add_argument("--n-transforms", type=int, default=None)
    ap.add_argument("--flow-epochs", type=int, default=100)
    ap.add_argument("--flow-patience", type=int, default=15)
    ap.add_argument("--flow-batch-size", type=int, default=256)
    ap.add_argument("--flow-lr", type=float, default=5e-4)

    ap.add_argument("--n-proposal-samples", type=int, default=5_000)
    ap.add_argument("--proposal-quantile-alpha", type=float, default=0.05)
    ap.add_argument("--proposal-pad-frac", type=float, default=0.25)
    ap.add_argument("--min-box-width-frac", type=float, default=0.2)

    ap.add_argument("--selection-strategies", default="topk")
    ap.add_argument("--selection-sizes", default="2000")
    ap.add_argument("--weight-temperature", type=float, default=1.0)
    ap.add_argument("--weight-ridge-frac", type=float, default=0.05)
    ap.add_argument("--selected-box-pad-frac", type=float, default=0.15)

    ap.add_argument("--nle-density-estimator", default="maf")
    ap.add_argument("--nle-sample-with", default="mcmc", choices=["mcmc"])
    ap.add_argument("--nle-mcmc-method", default="slice_np_vectorized")
    ap.add_argument("--nle-stop-after-epochs", type=int, default=20)
    ap.add_argument("--nle-max-num-epochs", type=int, default=200)
    ap.add_argument("--nle-batch-size", type=int, default=200)
    ap.add_argument("--nle-lr", type=float, default=5e-4)

    ap.add_argument("--n-posterior-samples", type=int, default=1_000)
    ap.add_argument("--c2st-folds", type=int, default=3)
    ap.add_argument("--c2st-max-epochs", type=int, default=100)
    ap.add_argument("--hist-bins", type=int, default=12)
    ap.add_argument("--conditional-bins", type=int, default=10)
    ap.add_argument("--include-context-baselines", action="store_true")
    ap.add_argument("--out-dir", default="pfn_testing/sbi/outputs/pfn_weighted_fixed_table_nle")
    args = ap.parse_args()

    pair_values = tuple(int(x.strip()) for x in args.pair.split(","))
    if len(pair_values) != 2:
        raise SystemExit("--pair must have two comma-separated indices")
    pair = (pair_values[0], pair_values[1])
    strategies = [x.strip() for x in args.selection_strategies.split(",") if x.strip()]
    selection_sizes = parse_int_list(args.selection_sizes)

    print(
        f"Simulating fixed table for {args.task}: "
        f"n_train={args.n_train}, n_val={args.n_val}, seed={args.seed}"
    )
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    task = data["task"]
    prior_low, prior_high = get_uniform_prior_bounds(task)

    t0 = time.perf_counter()
    pfn_model = fit_pfn_proposal_model(data, args)
    pfn_fit_seconds = time.perf_counter() - t0

    rows: list[dict[str, Any]] = []
    samples_to_save: dict[str, np.ndarray] = {}

    print("\n[2/3] Running observation-specific fixed-table localization")
    for obs in args.obs:
        print(f"\n[obs {obs}]")
        obs_x = task.get_observation(num_observation=obs).numpy().squeeze(0)
        reference = task.get_reference_posterior_samples(num_observation=obs).numpy()
        reference = reference[: args.n_posterior_samples]
        samples_to_save[f"obs{obs}_reference"] = reference
        samples_to_save[f"obs{obs}_x"] = obs_x

        t_obs = time.perf_counter()
        proposal_samples = sample_pfn_proposal(
            pfn_model,
            obs_x,
            args.n_proposal_samples,
        )
        proposal_seconds = time.perf_counter() - t_obs
        pfn_eval = proposal_samples[: args.n_posterior_samples]
        samples_to_save[f"obs{obs}_pfn_proposal"] = pfn_eval

        pfn_low, pfn_high = proposal_box(
            proposal_samples,
            prior_low,
            prior_high,
            args.proposal_quantile_alpha,
            args.proposal_pad_frac,
            args.min_box_width_frac,
        )
        pfn_box_volume = float(np.prod((pfn_high - pfn_low) / (prior_high - prior_low)))
        pfn_ref_cov = float(np.all((reference >= pfn_low) & (reference <= pfn_high), axis=1).mean())
        print(f"  PFN proposal box volume={pfn_box_volume:.4f}, ref coverage={pfn_ref_cov:.3f}")

        pfn_row = evaluate_method("pfn_proposal", pfn_eval, reference, pair, args)
        pfn_row.update(
            {
                "task": args.task,
                "obs": obs,
                "seed": args.seed,
                "selection_strategy": "none",
                "selection_size": 0,
                "training_simulations_total": args.n_train,
                "new_simulations": 0,
                "method_seconds": proposal_seconds,
                "pfn_fit_seconds_shared": pfn_fit_seconds,
                "pfn_box_volume_frac": pfn_box_volume,
                "pfn_ref_box_coverage": pfn_ref_cov,
                "local_box_volume_frac": np.nan,
                "local_ref_box_coverage": np.nan,
            }
        )
        rows.append(pfn_row)

        logw = proposal_log_weights(
            data["thetas_train"],
            proposal_samples,
            prior_low,
            prior_high,
            args.weight_ridge_frac,
        )
        for strategy in strategies:
            for n_select in selection_sizes:
                method = f"pfn_weighted_{strategy}{n_select}_nle"
                print(f"  {method}")
                theta_sel, x_sel, info = select_fixed_table_rows(
                    data["thetas_train"],
                    data["xs_train"],
                    logw,
                    n_select,
                    strategy,
                    args.weight_temperature,
                    args.seed + 1_000 * obs + n_select,
                )
                box_low, box_high = local_box_from_theta(
                    theta_sel,
                    prior_low,
                    prior_high,
                    args.selected_box_pad_frac,
                    args.min_box_width_frac,
                )
                box_volume = float(np.prod((box_high - box_low) / (prior_high - prior_low)))
                ref_cov = float(np.all((reference >= box_low) & (reference <= box_high), axis=1).mean())
                t_method = time.perf_counter()
                samples = train_fixed_table_nle(
                    task,
                    theta_sel,
                    x_sel,
                    box_low,
                    box_high,
                    obs_x,
                    args,
                )
                method_seconds = time.perf_counter() - t_method
                samples_to_save[f"obs{obs}_{method}"] = samples

                row = evaluate_method(method, samples, reference, pair, args)
                row.update(
                    {
                        "task": args.task,
                        "obs": obs,
                        "seed": args.seed,
                        "selection_strategy": strategy,
                        "selection_size": n_select,
                        "training_simulations_total": args.n_train,
                        "new_simulations": 0,
                        "method_seconds": method_seconds,
                        "pfn_fit_seconds_shared": pfn_fit_seconds,
                        "pfn_box_volume_frac": pfn_box_volume,
                        "pfn_ref_box_coverage": pfn_ref_cov,
                        "local_box_volume_frac": box_volume,
                        "local_ref_box_coverage": ref_cov,
                        **info,
                    }
                )
                rows.append(row)
                print(
                    f"    joint={row['joint_c2st']:.4f} "
                    f"marg={row['marginal_c2st']:.4f} "
                    f"rank={row['rank_c2st']:.4f} "
                    f"local_cov={ref_cov:.3f} "
                    f"unique={info['selected_unique_count']:.0f}"
                )

    print("\n[3/3] Aggregating")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_tag = "-".join(str(o) for o in args.obs)
    size_tag = "-".join(str(s) for s in selection_sizes)
    strategy_tag = "-".join(strategies)
    stem = (
        f"{args.task}_obs{obs_tag}_n{args.n_train}_"
        f"{strategy_tag}{size_tag}_ps{args.n_posterior_samples}_s{args.seed}"
    )
    details_path = out_dir / f"{stem}_details.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    json_path = out_dir / f"{stem}.json"
    npz_path = out_dir / f"{stem}.npz"

    write_csv(details_path, rows)

    summary_rows: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary_rows.append(
            {
                "method": method.replace("pfn_weighted_", "PFN-weighted "),
                "training_simulations_total": args.n_train,
                "training_simulations_per_obs": args.n_train / len(args.obs),
                "new_simulations": 0,
                "joint_mean": mean_std([float(r["joint_c2st"]) for r in method_rows])[0],
                "joint_std_obs": mean_std([float(r["joint_c2st"]) for r in method_rows])[1],
                "marginal_mean": mean_std([float(r["marginal_c2st"]) for r in method_rows])[0],
                "marginal_std_obs": mean_std([float(r["marginal_c2st"]) for r in method_rows])[1],
                "rank_mean": mean_std([float(r["rank_c2st"]) for r in method_rows])[0],
                "rank_std_obs": mean_std([float(r["rank_c2st"]) for r in method_rows])[1],
                "pair_mean": mean_std([float(r["pair_copula_c2st"]) for r in method_rows])[0],
                "hist_tv_mean": mean_std([float(r["pair_hist_tv"]) for r in method_rows])[0],
                "local_ref_box_coverage_mean": mean_std(
                    [
                        float(r["local_ref_box_coverage"])
                        for r in method_rows
                        if np.isfinite(float(r["local_ref_box_coverage"]))
                    ]
                    or [np.nan],
                )[0],
                "local_box_volume_frac_mean": mean_std(
                    [
                        float(r["local_box_volume_frac"])
                        for r in method_rows
                        if np.isfinite(float(r["local_box_volume_frac"]))
                    ]
                    or [np.nan],
                )[0],
            }
        )
    summary_rows.extend(load_baselines(args, args.obs))
    write_csv(summary_path, summary_rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "summary": summary_rows,
                "details": rows,
            },
            f,
            indent=2,
            default=str,
        )
    np.savez(str(npz_path), **samples_to_save)

    print_comparison(summary_rows)
    print(f"\nWrote {details_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
