"""Fresh sbi baselines (NPE, NLE, NRE, FMPE) for SBI benchmarking.

Runs sbi's own inference methods directly on SBIBM tasks using current
sbi defaults (v0.23+). Uses the same training data pipeline as other
methods for fair comparison.

Usage:
    uv run python -m pfn_testing.sbi.sbi_baselines --task two_moons --method npe
    uv run python -m pfn_testing.sbi.sbi_baselines --all --method npe
    uv run python -m pfn_testing.sbi.sbi_baselines --all --all-methods
    uv run python -m pfn_testing.sbi.sbi_baselines --task slcp --method nle --sample-with vi
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pfn_testing.sbi.plotting import plot_diagnostics
from pfn_testing.sbi.sbibm_utils import (
    AVAILABLE_TASKS,
    evaluate_posterior,
    simulate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Method registry
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry: (sbi_class_name, init_kwarg_for_estimator, default_estimator, is_amortized)
METHOD_REGISTRY: dict[str, dict] = {
    "npe": {
        "class_name": "NPE",
        "estimator_kwarg": "density_estimator",
        "default_estimator": "maf",
        "amortized": True,
    },
    "nle": {
        "class_name": "NLE",
        "estimator_kwarg": "density_estimator",
        "default_estimator": "maf",
        "amortized": False,
    },
    "nre": {
        "class_name": "NRE",
        "estimator_kwarg": "classifier",
        "default_estimator": "resnet",
        "amortized": False,
    },
    "fmpe": {
        "class_name": "FMPE",
        "estimator_kwarg": "density_estimator",
        "default_estimator": "mlp",
        "amortized": True,
    },
}

ALL_METHODS = list(METHOD_REGISTRY.keys())


def _get_sbi_class(method: str):
    """Import and return the sbi inference class for the given method."""
    from sbi.inference import FMPE, NLE, NPE, NRE

    classes = {"NPE": NPE, "NLE": NLE, "NRE": NRE, "FMPE": FMPE}
    return classes[METHOD_REGISTRY[method]["class_name"]]


def _prior_to_device(prior, device: str):
    """Move an sbibm prior to the target device.

    sbibm priors are:
      - Independent(Uniform(low, high))       — most tasks
      - Independent(LogNormal(loc, scale))     — sir, lotka_volterra
      - MultivariateNormal(loc, scale_tril)    — gaussian_linear, bernoulli_glm*

    sbi requires the prior to live on the same device as training tensors.
    """
    if device == "cpu":
        return prior

    if isinstance(prior, torch.distributions.MultivariateNormal):
        return torch.distributions.MultivariateNormal(
            loc=prior.loc.to(device),
            scale_tril=prior.scale_tril.to(device),
        )

    # Independent(base_dist) — rebuild base on target device
    base = prior.base_dist
    if isinstance(base, torch.distributions.Uniform):
        new_base = torch.distributions.Uniform(
            base.low.to(device), base.high.to(device),
        )
    elif isinstance(base, torch.distributions.LogNormal):
        new_base = torch.distributions.LogNormal(
            base.loc.to(device), base.scale.to(device),
        )
    else:
        raise TypeError(
            f"Don't know how to move {type(base).__name__} prior to {device}"
        )
    return torch.distributions.Independent(
        new_base, prior.reinterpreted_batch_ndims,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    task_name: str = "two_moons"
    method: str = "npe"

    # Simulation
    n_train: int = 10_000
    n_val: int = 2_000
    seed: int = 42

    # Network architecture (None = method default)
    density_estimator: str | None = None
    embedding_net_layers: list[int] | None = None  # e.g. [128, 64] for MLP
    pretrain_embedding: bool = False  # supervised pre-training on θ prediction

    # Training (sbi defaults)
    training_batch_size: int = 200
    stop_after_epochs: int = 20
    learning_rate: float = 5e-4

    # Sampling for NLE/NRE
    sample_with: str = "mcmc"
    mcmc_method: str = "slice_np_vectorized"

    # Evaluation
    n_posterior_samples: int = 10_000
    n_reference_observations: int = 10

    @property
    def output_dir(self) -> Path:
        suffix = f"sbi_{self.method}_n{self.n_train}"
        if self.embedding_net_layers:
            suffix += f"_emb{'_'.join(str(x) for x in self.embedding_net_layers)}"
            if self.pretrain_embedding:
                suffix += "_pretrained"
        if self.seed != 42:
            suffix += f"_s{self.seed}"
        return Path("pfn_testing/sbi/outputs") / self.task_name / suffix


# ═══════════════════════════════════════════════════════════════════════════════
# Run single task
# ═══════════════════════════════════════════════════════════════════════════════

def run_task(cfg: Config) -> dict:
    """Run the full sbi inference pipeline on a single sbibm task.

    Returns:
        {"task_name", "method", "dim_theta", "dim_x",
         "c2st_sbi_{method}": list[float]}
    """
    method_info = METHOD_REGISTRY[cfg.method]
    out = cfg.output_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"sbi {cfg.method.upper()} Baseline: {cfg.task_name}")
    print("=" * 60)

    # ── 1. Simulate ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Simulating {cfg.n_train + cfg.n_val} samples...")
    data = simulate(cfg.task_name, cfg.n_train, cfg.n_val, cfg.seed)
    task = data["task"]
    dim_theta, dim_x = data["dim_theta"], data["dim_x"]
    print(f"  Task: {cfg.task_name} (dim_theta={dim_theta}, dim_x={dim_x})")
    print(f"  Train: {data['thetas_train'].shape}, Val: {data['thetas_val'].shape}")

    # ── 2. Build inference object ────────────────────────────────────────────
    estimator = cfg.density_estimator or method_info["default_estimator"]
    print(f"\n[2/4] Building sbi {cfg.method.upper()} "
          f"(estimator={estimator})...")

    sbi_class = _get_sbi_class(cfg.method)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the prior distribution for sbi (must be on same device)
    prior_dist = _prior_to_device(task.get_prior_dist(), device)

    # Build embedding net if configured
    if cfg.embedding_net_layers:
        from sbi.neural_nets import posterior_nn

        layers = []
        in_dim = dim_x
        for width in cfg.embedding_net_layers:
            layers.extend([torch.nn.Linear(in_dim, width), torch.nn.ReLU()])
            in_dim = width
        embedding_net = torch.nn.Sequential(*layers)
        emb_out_dim = cfg.embedding_net_layers[-1]
        print(f"  Embedding net: {' -> '.join(str(d) for d in [dim_x] + cfg.embedding_net_layers)}")

        # Supervised pre-training: train embedding + linear head to predict θ
        if cfg.pretrain_embedding:
            print("\n  Pre-training embedding net (predict θ from x)...")
            pretrain_head = torch.nn.Linear(emb_out_dim, dim_theta)
            pretrain_model = torch.nn.Sequential(
                embedding_net, pretrain_head,
            ).to(device)

            x_train_t = torch.tensor(
                data["xs_train"], dtype=torch.float32, device=device,
            )
            theta_train_t = torch.tensor(
                data["thetas_train"], dtype=torch.float32, device=device,
            )
            x_val_t = torch.tensor(
                data["xs_val"], dtype=torch.float32, device=device,
            )
            theta_val_t = torch.tensor(
                data["thetas_val"], dtype=torch.float32, device=device,
            )

            pretrain_opt = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)
            best_val_loss = float("inf")
            no_improve = 0

            for epoch in range(500):
                # Mini-batch training
                pretrain_model.train()
                perm = torch.randperm(len(x_train_t), device=device)
                epoch_loss = 0.0
                n_batches = 0
                for i in range(0, len(x_train_t), cfg.training_batch_size):
                    batch_idx = perm[i:i + cfg.training_batch_size]
                    pred = pretrain_model(x_train_t[batch_idx])
                    loss = torch.nn.functional.mse_loss(pred, theta_train_t[batch_idx])
                    pretrain_opt.zero_grad()
                    loss.backward()
                    pretrain_opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                # Validation
                pretrain_model.eval()
                with torch.no_grad():
                    val_pred = pretrain_model(x_val_t)
                    val_loss = torch.nn.functional.mse_loss(
                        val_pred, theta_val_t,
                    ).item()

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= 30:
                    print(f"  Pre-training converged at epoch {epoch} "
                          f"(val MSE: {best_val_loss:.6f})")
                    break
            else:
                print(f"  Pre-training finished 500 epochs "
                      f"(val MSE: {best_val_loss:.6f})")

            # Move embedding_net back to cpu (sbi will move it)
            embedding_net = embedding_net.cpu()

        density_estimator_build_fn = posterior_nn(
            model=estimator,
            embedding_net=embedding_net,
            z_score_x="none",  # embedding net handles input
        )

        init_kwargs = {
            "prior": prior_dist,
            method_info["estimator_kwarg"]: density_estimator_build_fn,
            "device": device,
        }
    else:
        init_kwargs = {
            "prior": prior_dist,
            method_info["estimator_kwarg"]: estimator,
            "device": device,
        }

    inference = sbi_class(**init_kwargs)

    # ── 3. Train ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training (batch_size={cfg.training_batch_size}, "
          f"stop_after_epochs={cfg.stop_after_epochs})...")

    theta_tensor = torch.tensor(data["thetas_train"], dtype=torch.float32)
    x_tensor = torch.tensor(data["xs_train"], dtype=torch.float32)

    inference.append_simulations(theta_tensor, x_tensor)

    train_kwargs = {
        "training_batch_size": cfg.training_batch_size,
        "stop_after_epochs": cfg.stop_after_epochs,
        "learning_rate": cfg.learning_rate,
    }
    inference.train(**train_kwargs)

    # ── Build posterior ──────────────────────────────────────────────────────
    build_kwargs: dict = {}
    if not method_info["amortized"]:
        # NLE/NRE need MCMC or VI sampling
        build_kwargs["sample_with"] = cfg.sample_with
        if cfg.sample_with == "mcmc":
            build_kwargs["mcmc_method"] = cfg.mcmc_method

    posterior = inference.build_posterior(**build_kwargs)

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating on {cfg.n_reference_observations} "
          "sbibm observations...")

    def sample_fn(x_obs_np: np.ndarray) -> np.ndarray:
        x_obs_tensor = torch.tensor(
            x_obs_np, dtype=torch.float32,
        ).to(device)
        samples = posterior.sample(
            (cfg.n_posterior_samples,), x=x_obs_tensor,
        )
        return samples.cpu().numpy()

    c2st_scores = evaluate_posterior(
        task, sample_fn, cfg.n_reference_observations, cfg.n_posterior_samples,
    )

    # ── Diagnostics for obs 1 ────────────────────────────────────────────────
    x_obs_1 = task.get_observation(num_observation=1).numpy().squeeze(0)
    ref_1 = task.get_reference_posterior_samples(num_observation=1).numpy()
    post_1 = sample_fn(x_obs_1)

    plot_diagnostics(
        post_1, ref_1,
        {"train_loss": [], "val_loss": []},
        f"sbi {cfg.method.upper()}",
        str(out / "plots" / f"sbi_{cfg.method}_diagnostics.png"),
    )

    # ── Save results ─────────────────────────────────────────────────────────
    c2st_key = f"c2st_sbi_{cfg.method}"
    results = {
        "task_name": cfg.task_name,
        "method": cfg.method,
        "dim_theta": dim_theta,
        "dim_x": dim_x,
        c2st_key: c2st_scores,
    }
    np.savez(
        str(out / "results" / "results.npz"),
        **{k: np.array(v) if isinstance(v, list) else v
           for k, v in results.items()},
    )

    # ── Print summary ────────────────────────────────────────────────────────
    mean, std = np.mean(c2st_scores), np.std(c2st_scores)
    print(f"\n  sbi {cfg.method.upper()} C2ST: {mean:.4f} +/- {std:.4f}")
    print(f"  Outputs: {out}/")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results_list: list[dict]) -> None:
    """Print a cross-task summary table."""
    if not results_list:
        return

    method = results_list[0]["method"]
    c2st_key = f"c2st_sbi_{method}"

    print("\n" + "=" * 60)
    print(f"sbi {method.upper()} Benchmark Summary")
    print("=" * 60)
    print(f"\n{'Task':<28s} {'dim_θ':>5s} {'dim_x':>5s} "
          f"{'C2ST':>16s}")
    print("-" * 60)
    for r in results_list:
        mean = np.mean(r[c2st_key])
        std = np.std(r[c2st_key])
        print(
            f"{r['task_name']:<28s} {r['dim_theta']:5d} {r['dim_x']:5d} "
            f"{mean:6.4f}±{std:.4f}"
        )
    print()
    print("(C2ST closer to 0.5 = better)")


def run_benchmark(
    task_names: list[str],
    method: str,
    cfg_overrides: dict | None = None,
) -> list[dict]:
    """Run a sbi method on multiple tasks and save summary."""
    all_results = []

    for i, task_name in enumerate(task_names, 1):
        print(f"\n{'#' * 60}")
        print(f"# Task {i}/{len(task_names)}: {task_name}")
        print(f"{'#' * 60}\n")

        cfg = Config(task_name=task_name, method=method)
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)

        results = run_task(cfg)
        all_results.append(results)

    print_summary(all_results)

    out_dir = Path("pfn_testing/sbi/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    c2st_key = f"c2st_sbi_{method}"
    np.savez(
        str(out_dir / f"sbi_{method}_summary.npz"),
        **{r["task_name"]: np.array(r[c2st_key]) for r in all_results},
    )

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fresh sbi baselines for SBI benchmarking",
    )
    parser.add_argument(
        "--task", type=str, default="two_moons",
        help=f"SBIBM task name (choices: {', '.join(AVAILABLE_TASKS)})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all available SBIBM tasks",
    )
    parser.add_argument(
        "--method", type=str, default="npe",
        choices=ALL_METHODS,
        help=f"sbi method (choices: {', '.join(ALL_METHODS)})",
    )
    parser.add_argument(
        "--all-methods", action="store_true",
        help="Run all available sbi methods",
    )
    parser.add_argument(
        "--n-train", type=int, default=10_000,
        help="Number of training simulations (default: 10000)",
    )
    parser.add_argument(
        "--n-obs", type=int, default=None,
        help="Number of reference observations for evaluation (default: 10)",
    )
    parser.add_argument(
        "--n-posterior-samples", type=int, default=None,
        help="Posterior samples per observation for C2ST (default: 10000)",
    )
    parser.add_argument(
        "--density-estimator", type=str, default=None,
        help="Override density estimator (e.g., nsf, maf, mdn, resnet, mlp)",
    )
    parser.add_argument(
        "--sample-with", type=str, default="mcmc",
        help="Sampling method for NLE/NRE (mcmc, vi, rejection)",
    )
    parser.add_argument(
        "--mcmc-method", type=str, default="slice_np_vectorized",
        help="MCMC method when --sample-with=mcmc",
    )
    parser.add_argument(
        "--embedding-net", type=int, nargs="+", default=None,
        help="MLP embedding net layer widths (e.g., --embedding-net 128 64)",
    )
    parser.add_argument(
        "--pretrain-embedding", action="store_true",
        help="Supervised pre-training: train embedding net to predict theta before end-to-end",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    overrides: dict = {"n_train": args.n_train, "seed": args.seed}
    if args.n_obs is not None:
        overrides["n_reference_observations"] = args.n_obs
    if args.n_posterior_samples is not None:
        overrides["n_posterior_samples"] = args.n_posterior_samples
    if args.density_estimator is not None:
        overrides["density_estimator"] = args.density_estimator
    overrides["sample_with"] = args.sample_with
    overrides["mcmc_method"] = args.mcmc_method
    if args.embedding_net is not None:
        overrides["embedding_net_layers"] = args.embedding_net
    overrides["pretrain_embedding"] = args.pretrain_embedding

    task_names = AVAILABLE_TASKS if args.all else [args.task]
    methods = ALL_METHODS if args.all_methods else [args.method]

    for method in methods:
        run_benchmark(task_names, method, overrides)


if __name__ == "__main__":
    main()
