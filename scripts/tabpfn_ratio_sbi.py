"""TabPFN likelihood-to-evidence ratio estimator for SBI.

This prototype uses TabPFN as an in-context classifier over full
``(theta, x)`` pairs. Positive examples are joint simulator pairs
``theta ~ p(theta), x ~ p(x | theta)``. Negative examples are shuffled
pairs approximating ``theta ~ p(theta), x ~ p(x)``. With balanced classes,
the classifier odds estimate

    p(y=1 | theta, x) / p(y=0 | theta, x)
      ~= p(theta, x) / (p(theta) p(x))
       = p(x | theta) / p(x).

For an observation x_o, posterior samples are drawn by importance-resampling
prior candidates with weights proportional to these odds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import compute_c2st, simulate  # noqa: E402


def create_classifier(model_version: str, device: str) -> TabPFNClassifier:
    if model_version not in {"v2", "v2.5"}:
        raise ValueError("--model-version must be v2 or v2.5")
    version = ModelVersion.V2 if model_version == "v2" else ModelVersion.V2_5
    if device == "auto":
        return TabPFNClassifier.create_default_for_version(version, n_estimators=1)
    return TabPFNClassifier.create_default_for_version(
        version,
        n_estimators=1,
        device=device,
    )


def pair_features(theta: np.ndarray, x: np.ndarray, feature_map: str = "raw") -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.repeat(x.reshape(1, -1), len(theta), axis=0)
    raw = np.concatenate([theta, x], axis=1)
    if feature_map == "raw":
        return raw.astype(np.float32, copy=False)
    if feature_map == "poly2":
        outer = (theta[:, :, None] * x[:, None, :]).reshape(len(theta), -1)
        return np.concatenate([raw, theta**2, x**2, outer], axis=1).astype(
            np.float32,
            copy=False,
        )
    if feature_map == "interactions":
        outer = (theta[:, :, None] * x[:, None, :]).reshape(len(theta), -1)
        return np.concatenate([raw, outer], axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unknown feature map: {feature_map}")


def standardize_features(
    features: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    if mean is None or std is None:
        return features
    return ((features - mean) / std).astype(np.float32, copy=False)


def make_ratio_context(
    thetas: np.ndarray,
    xs: np.ndarray,
    n_context_pairs: int,
    seed: int,
    negative_strategy: str,
    feature_map: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(n_context_pairs, len(thetas), len(xs))
    idx = rng.choice(len(thetas), size=n, replace=False)

    theta_pos = thetas[idx]
    x_pos = xs[idx]

    if negative_strategy == "shuffle_x":
        perm = idx.copy()
        while True:
            rng.shuffle(perm)
            if np.all(perm != idx) or n < 2:
                break
        theta_neg = theta_pos
        x_neg = xs[perm]
    elif negative_strategy == "independent":
        theta_idx = rng.choice(len(thetas), size=n, replace=True)
        x_idx = rng.choice(len(xs), size=n, replace=True)
        theta_neg = thetas[theta_idx]
        x_neg = xs[x_idx]
    else:
        raise ValueError(f"Unknown negative strategy: {negative_strategy}")

    z = np.concatenate(
        [
            pair_features(theta_pos, x_pos, feature_map),
            pair_features(theta_neg, x_neg, feature_map),
        ],
        axis=0,
    )
    y = np.concatenate(
        [
            np.ones(n, dtype=np.int64),
            np.zeros(n, dtype=np.int64),
        ],
        axis=0,
    )
    order = rng.permutation(len(y))
    return z[order], y[order]


def prior_candidates(task: object, n_candidates: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    prior = task.get_prior()
    return prior(num_samples=n_candidates).numpy().astype(np.float32)


def probability_to_log_weight(
    prob_joint: np.ndarray,
    logit_temperature: float,
) -> np.ndarray:
    p = np.clip(np.asarray(prob_joint, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(p) - np.log1p(-p)
    if logit_temperature <= 0.0:
        raise ValueError("--logit-temperature must be positive")
    return logits / logit_temperature


def normalize_log_weights(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    shifted = log_w - np.max(log_w)
    w = np.exp(shifted)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        w = np.full_like(w, 1.0 / len(w), dtype=np.float64)
    else:
        w = w / total
    ess = 1.0 / float(np.sum(w**2))
    return w, ess


def score_candidates(
    clf: TabPFNClassifier,
    candidates: np.ndarray,
    x_obs: np.ndarray,
    batch_size: int,
    logit_temperature: float,
    feature_map: str,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    probs = []
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        z = pair_features(candidates[start:end], x_obs, feature_map)
        z = standardize_features(z, feature_mean, feature_std)
        batch_probs = clf.predict_proba(z)[:, 1]
        probs.append(np.asarray(batch_probs, dtype=np.float64))
    prob_joint = np.concatenate(probs, axis=0)
    log_w = probability_to_log_weight(prob_joint, logit_temperature)
    weights, ess = normalize_log_weights(log_w)
    return prob_joint, weights, ess


def log_ratio_for_thetas(
    clf: TabPFNClassifier,
    thetas: np.ndarray,
    x_obs: np.ndarray,
    feature_map: str,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    logit_temperature: float,
) -> np.ndarray:
    z = pair_features(thetas, x_obs, feature_map)
    z = standardize_features(z, feature_mean, feature_std)
    prob = clf.predict_proba(z)[:, 1]
    return probability_to_log_weight(prob, logit_temperature)


def resample_posterior(
    candidates: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=n_samples, replace=True, p=weights)
    return candidates[idx], idx


def prior_box_bounds(task: object) -> tuple[np.ndarray, np.ndarray]:
    prior_dist = task.get_prior_dist()
    if isinstance(prior_dist, torch.distributions.Independent) and isinstance(
        prior_dist.base_dist,
        torch.distributions.Uniform,
    ):
        return (
            prior_dist.base_dist.low.detach().cpu().numpy().astype(np.float32),
            prior_dist.base_dist.high.detach().cpu().numpy().astype(np.float32),
        )
    if isinstance(prior_dist, torch.distributions.Uniform):
        return (
            prior_dist.low.detach().cpu().numpy().astype(np.float32),
            prior_dist.high.detach().cpu().numpy().astype(np.float32),
        )
    raise TypeError(
        "MCMC sampler currently supports box-uniform priors only. "
        f"Got {type(prior_dist).__name__}."
    )


def mcmc_posterior(
    clf: TabPFNClassifier,
    candidates: np.ndarray,
    init_weights: np.ndarray,
    x_obs: np.ndarray,
    task: object,
    n_samples: int,
    seed: int,
    feature_map: str,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    logit_temperature: float,
    n_chains: int,
    burnin: int,
    thin: int,
    step_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    low, high = prior_box_bounds(task)
    width = high - low
    proposal_std = step_scale * width

    init_idx = rng.choice(len(candidates), size=n_chains, replace=True, p=init_weights)
    current = candidates[init_idx].astype(np.float32, copy=True)
    current_log_ratio = log_ratio_for_thetas(
        clf,
        current,
        x_obs,
        feature_map,
        feature_mean,
        feature_std,
        logit_temperature,
    )

    draws: list[np.ndarray] = []
    accepted = 0
    proposed_valid = 0
    total_steps = burnin + int(np.ceil(n_samples / n_chains)) * thin

    for step in range(total_steps):
        proposal = current + rng.normal(scale=proposal_std, size=current.shape).astype(np.float32)
        valid = np.all((proposal >= low) & (proposal <= high), axis=1)
        proposal_log_ratio = np.full(n_chains, -np.inf, dtype=np.float64)
        if np.any(valid):
            proposed_valid += int(valid.sum())
            proposal_log_ratio[valid] = log_ratio_for_thetas(
                clf,
                proposal[valid],
                x_obs,
                feature_map,
                feature_mean,
                feature_std,
                logit_temperature,
            )

        log_alpha = proposal_log_ratio - current_log_ratio
        accept = np.log(rng.random(n_chains)) < log_alpha
        if np.any(accept):
            current[accept] = proposal[accept]
            current_log_ratio[accept] = proposal_log_ratio[accept]
            accepted += int(accept.sum())

        if step >= burnin and (step - burnin) % thin == 0:
            draws.append(current.copy())

    samples = np.concatenate(draws, axis=0)[:n_samples]
    total_proposals = total_steps * n_chains
    info = {
        "mcmc_acceptance": accepted / max(total_proposals, 1),
        "mcmc_valid_proposal_frac": proposed_valid / max(total_proposals, 1),
        "mcmc_steps": float(total_steps),
        "mcmc_chains": float(n_chains),
    }
    return samples, info


def batched_local_posterior(
    clf: TabPFNClassifier,
    candidates: np.ndarray,
    init_weights: np.ndarray,
    x_obs: np.ndarray,
    task: object,
    n_samples: int,
    seed: int,
    feature_map: str,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    logit_temperature: float,
    n_chains: int,
    burnin_rounds: int,
    keep_rounds: int,
    proposals_per_chain: int,
    step_scale: float,
    step_decay: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Approximate local-proposal sampler with batched TabPFN scoring.

    Each round proposes K local moves for every chain, scores all K*C proposals
    in one forward pass, and chooses the next state from {current, proposals}
    by a local softmax over the ratio scores. This is not an exact MCMC kernel;
    it is a fast diagnostic sampler for whether the TabPFN ratio landscape can
    guide particles into plausible posterior regions.
    """
    rng = np.random.default_rng(seed)
    low, high = prior_box_bounds(task)
    width = high - low

    init_idx = rng.choice(len(candidates), size=n_chains, replace=True, p=init_weights)
    current = candidates[init_idx].astype(np.float32, copy=True)
    current_log_ratio = log_ratio_for_thetas(
        clf,
        current,
        x_obs,
        feature_map,
        feature_mean,
        feature_std,
        logit_temperature,
    )

    draws: list[np.ndarray] = []
    moved = 0
    valid_props = 0
    total_props = 0
    total_rounds = burnin_rounds + keep_rounds

    for round_idx in range(total_rounds):
        round_scale = step_scale * (step_decay ** max(round_idx - burnin_rounds, 0))
        proposal_std = round_scale * width
        proposals = (
            current[:, None, :]
            + rng.normal(
                scale=proposal_std,
                size=(n_chains, proposals_per_chain, current.shape[1]),
            ).astype(np.float32)
        )
        flat = proposals.reshape(-1, current.shape[1])
        valid = np.all((flat >= low) & (flat <= high), axis=1)
        proposal_log_ratio = np.full(len(flat), -np.inf, dtype=np.float64)
        if np.any(valid):
            valid_props += int(valid.sum())
            proposal_log_ratio[valid] = log_ratio_for_thetas(
                clf,
                flat[valid],
                x_obs,
                feature_map,
                feature_mean,
                feature_std,
                logit_temperature,
            )
        total_props += len(flat)

        proposal_log_ratio = proposal_log_ratio.reshape(n_chains, proposals_per_chain)
        all_states = np.concatenate([current[:, None, :], proposals], axis=1)
        all_scores = np.concatenate([current_log_ratio[:, None], proposal_log_ratio], axis=1)
        all_scores = all_scores - np.max(all_scores, axis=1, keepdims=True)
        choice_weights = np.exp(all_scores)
        choice_weights /= choice_weights.sum(axis=1, keepdims=True)

        choices = np.empty(n_chains, dtype=np.int64)
        for i in range(n_chains):
            choices[i] = rng.choice(proposals_per_chain + 1, p=choice_weights[i])
        moved += int(np.sum(choices != 0))
        current = all_states[np.arange(n_chains), choices]
        chosen_scores = np.concatenate([current_log_ratio[:, None], proposal_log_ratio], axis=1)
        current_log_ratio = chosen_scores[np.arange(n_chains), choices]

        if round_idx >= burnin_rounds:
            draws.append(current.copy())

    samples = np.concatenate(draws, axis=0)
    if len(samples) < n_samples:
        extra_idx = rng.choice(len(samples), size=n_samples - len(samples), replace=True)
        samples = np.concatenate([samples, samples[extra_idx]], axis=0)
    samples = samples[:n_samples]
    info = {
        "batched_local_move_frac": moved / max(n_chains * total_rounds, 1),
        "batched_local_valid_proposal_frac": valid_props / max(total_props, 1),
        "batched_local_rounds": float(total_rounds),
        "batched_local_chains": float(n_chains),
        "batched_local_proposals_per_chain": float(proposals_per_chain),
    }
    return samples, info


def pooled_ranks(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    pooled = np.concatenate([a[:n], b[:n]], axis=0)
    ranks = np.empty_like(pooled, dtype=np.float64)
    denom = pooled.shape[0] + 1.0
    for d in range(pooled.shape[1]):
        order = np.argsort(pooled[:, d], kind="mergesort")
        ranks[order, d] = (np.arange(pooled.shape[0]) + 1.0) / denom
    return ranks[:n], ranks[n:]


def c2st_decomposition(
    samples: np.ndarray,
    reference: np.ndarray,
    c2st_folds: int,
    c2st_max_epochs: int,
    seed: int,
) -> dict[str, Any]:
    n = min(len(samples), len(reference))
    model = np.asarray(samples[:n], dtype=np.float64)
    ref = np.asarray(reference[:n], dtype=np.float64)

    joint = compute_c2st(
        model,
        ref,
        n_folds=c2st_folds,
        max_epochs=c2st_max_epochs,
        seed=seed,
    )
    marginal_per_dim = []
    for d in range(model.shape[1]):
        marginal_per_dim.append(
            compute_c2st(
                model[:, [d]],
                ref[:, [d]],
                n_folds=c2st_folds,
                max_epochs=c2st_max_epochs,
                seed=seed + 100 + d,
            )
        )
    model_rank, ref_rank = pooled_ranks(model, ref)
    rank = compute_c2st(
        model_rank,
        ref_rank,
        n_folds=c2st_folds,
        max_epochs=c2st_max_epochs,
        seed=seed + 10_000,
    )
    return {
        "joint_c2st": float(joint),
        "marginal_c2st": float(np.mean(marginal_per_dim)),
        "rank_c2st": float(rank),
        "marginal_c2st_per_dim": marginal_per_dim,
    }


def weighted_mean_and_cov(candidates: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(candidates, axis=0, weights=weights)
    centered = candidates - mean
    cov = (centered * weights[:, None]).T @ centered
    return mean, cov


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_obs_list(obs_arg: str, n_obs: int) -> list[int]:
    if obs_arg == "first":
        return list(range(1, n_obs + 1))
    return [int(x.strip()) for x in obs_arg.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"])
    ap.add_argument("--context-pairs", type=int, default=3_000)
    ap.add_argument("--negative-strategy", default="shuffle_x", choices=["shuffle_x", "independent"])
    ap.add_argument("--feature-map", default="raw", choices=["raw", "interactions", "poly2"])
    ap.add_argument("--standardize-pair-features", action="store_true")
    ap.add_argument("--n-candidates", type=int, default=50_000)
    ap.add_argument("--candidate-batch-size", type=int, default=2_000)
    ap.add_argument("--n-posterior-samples", type=int, default=1_000)
    ap.add_argument("--sampler", default="importance", choices=["importance", "mcmc", "batched_local"])
    ap.add_argument("--mcmc-chains", type=int, default=100)
    ap.add_argument("--mcmc-burnin", type=int, default=200)
    ap.add_argument("--mcmc-thin", type=int, default=2)
    ap.add_argument("--mcmc-step-scale", type=float, default=0.08)
    ap.add_argument("--local-chains", type=int, default=200)
    ap.add_argument("--local-burnin-rounds", type=int, default=10)
    ap.add_argument("--local-keep-rounds", type=int, default=10)
    ap.add_argument("--local-proposals-per-chain", type=int, default=16)
    ap.add_argument("--local-step-scale", type=float, default=0.05)
    ap.add_argument("--local-step-decay", type=float, default=0.95)
    ap.add_argument("--n-obs", type=int, default=10)
    ap.add_argument(
        "--obs",
        default="first",
        help="'first' for 1..n_obs, or comma-separated observation ids",
    )
    ap.add_argument("--logit-temperature", type=float, default=2.0)
    ap.add_argument("--c2st-folds", type=int, default=3)
    ap.add_argument("--c2st-max-epochs", type=int, default=100)
    ap.add_argument("--save-candidate-scores", action="store_true")
    ap.add_argument("--out-dir", default="pfn_testing/sbi/outputs/tabpfn_ratio")
    args = ap.parse_args()

    print(
        f"Simulating {args.task}: n_train={args.n_train}, n_val={args.n_val}, "
        f"seed={args.seed}"
    )
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    task = data["task"]

    print(
        "Building ratio context: "
        f"{args.context_pairs} joint + {args.context_pairs} shuffled pairs"
    )
    z_context, y_context = make_ratio_context(
        data["thetas_train"],
        data["xs_train"],
        args.context_pairs,
        args.seed,
        args.negative_strategy,
        args.feature_map,
    )
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    if args.standardize_pair_features:
        feature_mean = z_context.mean(axis=0)
        feature_std = z_context.std(axis=0) + 1e-6
        z_context = standardize_features(z_context, feature_mean, feature_std)
    print(f"  Context shape: {z_context.shape}; positive rate={y_context.mean():.3f}")

    print("Fitting TabPFN classifier")
    clf = create_classifier(args.model_version, args.device)
    clf.fit(z_context, y_context)

    print(f"Sampling {args.n_candidates} prior candidates")
    candidates = prior_candidates(task, args.n_candidates, args.seed + 9_999)

    obs_ids = parse_obs_list(args.obs, args.n_obs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{args.task}_n{args.n_train}_ctx{args.context_pairs}_{args.feature_map}_"
        f"cand{args.n_candidates}_{args.sampler}_ps{args.n_posterior_samples}_s{args.seed}"
    )
    summary_rows: list[dict[str, Any]] = []
    samples_payload: dict[str, Any] = {
        "obs_ids": np.asarray(obs_ids, dtype=np.int64),
        "candidates_shape": np.asarray(candidates.shape, dtype=np.int64),
        "feature_mean": np.asarray([] if feature_mean is None else feature_mean),
        "feature_std": np.asarray([] if feature_std is None else feature_std),
    }
    if args.save_candidate_scores:
        samples_payload["candidates"] = candidates

    for obs_id in obs_ids:
        print(f"\nObservation {obs_id}: scoring candidates")
        x_obs = task.get_observation(num_observation=obs_id).numpy().squeeze(0)
        ref = task.get_reference_posterior_samples(num_observation=obs_id).numpy()
        ref = ref[: args.n_posterior_samples]

        prob_joint, weights, ess = score_candidates(
            clf,
            candidates,
            x_obs,
            args.candidate_batch_size,
            args.logit_temperature,
            args.feature_map,
            feature_mean,
            feature_std,
        )
        sampler_info: dict[str, float] = {}
        if args.sampler == "importance":
            post, resample_idx = resample_posterior(
                candidates,
                weights,
                args.n_posterior_samples,
                args.seed + obs_id,
            )
        elif args.sampler == "mcmc":
            post, sampler_info = mcmc_posterior(
                clf,
                candidates,
                weights,
                x_obs,
                task,
                args.n_posterior_samples,
                args.seed + obs_id,
                args.feature_map,
                feature_mean,
                feature_std,
                args.logit_temperature,
                args.mcmc_chains,
                args.mcmc_burnin,
                args.mcmc_thin,
                args.mcmc_step_scale,
            )
            resample_idx = np.asarray([], dtype=np.int64)
        else:
            post, sampler_info = batched_local_posterior(
                clf,
                candidates,
                weights,
                x_obs,
                task,
                args.n_posterior_samples,
                args.seed + obs_id,
                args.feature_map,
                feature_mean,
                feature_std,
                args.logit_temperature,
                args.local_chains,
                args.local_burnin_rounds,
                args.local_keep_rounds,
                args.local_proposals_per_chain,
                args.local_step_scale,
                args.local_step_decay,
            )
            resample_idx = np.asarray([], dtype=np.int64)
        decomp = c2st_decomposition(
            post,
            ref,
            args.c2st_folds,
            args.c2st_max_epochs,
            seed=args.seed + obs_id,
        )
        w_mean, w_cov = weighted_mean_and_cov(candidates, weights)
        ref_mean = ref.mean(axis=0)
        ref_cov = np.cov(ref, rowvar=False)
        cov_rel_fro = float(
            np.linalg.norm(w_cov - ref_cov, ord="fro")
            / (np.linalg.norm(ref_cov, ord="fro") + 1e-8)
        )

        row = {
            "task": args.task,
            "method": "tabpfn_ratio",
            "obs": obs_id,
            "n_train": args.n_train,
            "context_pairs": args.context_pairs,
            "feature_map": args.feature_map,
            "sampler": args.sampler,
            "n_candidates": args.n_candidates,
            "n_posterior_samples": args.n_posterior_samples,
            "ess": float(ess),
            "ess_frac": float(ess / len(candidates)),
            "max_weight": float(weights.max()),
            "mean_prob_joint": float(prob_joint.mean()),
            "max_prob_joint": float(prob_joint.max()),
            "mean_rmse": float(np.sqrt(np.mean((w_mean - ref_mean) ** 2))),
            "cov_rel_fro": cov_rel_fro,
            **sampler_info,
            **{k: v for k, v in decomp.items() if k != "marginal_c2st_per_dim"},
        }
        summary_rows.append(row)
        samples_payload[f"obs{obs_id}_samples"] = post
        samples_payload[f"obs{obs_id}_reference"] = ref
        samples_payload[f"obs{obs_id}_resample_idx"] = resample_idx
        if args.save_candidate_scores:
            samples_payload[f"obs{obs_id}_prob_joint"] = prob_joint
            samples_payload[f"obs{obs_id}_weights"] = weights

        print(
            f"  C2ST joint={row['joint_c2st']:.4f} "
            f"marginal={row['marginal_c2st']:.4f} rank={row['rank_c2st']:.4f} "
            f"ESS={row['ess']:.1f} ({row['ess_frac']:.3%})"
        )

    mean_row = {
        "task": args.task,
        "method": "tabpfn_ratio",
        "obs": "mean",
        "n_train": args.n_train,
        "context_pairs": args.context_pairs,
        "feature_map": args.feature_map,
        "sampler": args.sampler,
        "n_candidates": args.n_candidates,
        "n_posterior_samples": args.n_posterior_samples,
    }
    numeric_keys = [
        "joint_c2st",
        "marginal_c2st",
        "rank_c2st",
        "ess",
        "ess_frac",
        "max_weight",
        "mean_prob_joint",
        "max_prob_joint",
        "mean_rmse",
        "cov_rel_fro",
        "mcmc_acceptance",
        "mcmc_valid_proposal_frac",
        "mcmc_steps",
        "mcmc_chains",
        "batched_local_move_frac",
        "batched_local_valid_proposal_frac",
        "batched_local_rounds",
        "batched_local_chains",
        "batched_local_proposals_per_chain",
    ]
    for key in numeric_keys:
        vals = [float(r[key]) for r in summary_rows if key in r]
        if vals:
            mean_row[key] = float(np.mean(vals))

    all_rows = summary_rows + [mean_row]
    summary_csv = out_dir / f"{stem}_summary.csv"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    write_csv(summary_csv, all_rows)
    np.savez(str(npz_path), **samples_payload)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": all_rows}, f, indent=2, default=str)

    print("\nTabPFN Ratio Summary")
    print("-" * 78)
    print(f"{'obs':>5} {'joint':>8} {'marg':>8} {'rank':>8} {'ESS%':>8} {'cov':>8}")
    for row in all_rows:
        print(
            f"{str(row['obs']):>5} "
            f"{float(row['joint_c2st']):>8.4f} "
            f"{float(row['marginal_c2st']):>8.4f} "
            f"{float(row['rank_c2st']):>8.4f} "
            f"{100.0 * float(row['ess_frac']):>8.3f} "
            f"{float(row['cov_rel_fro']):>8.4f}"
        )
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
