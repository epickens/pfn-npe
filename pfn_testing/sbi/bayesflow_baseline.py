"""BayesFlow NPE baseline for SBI benchmarking.

Pipeline:
    Simulator(theta) -> data x
    -> CouplingFlow q(theta | x)  [no summary network: x is fixed-dim]
    -> Posterior samples

Equivalent to amortized NPE using BayesFlow's CouplingFlow.

Usage:
    uv run python -m pfn_testing.sbi.bayesflow_baseline --task two_moons
    uv run python -m pfn_testing.sbi.bayesflow_baseline --task slcp --n-train 20000
    uv run python -m pfn_testing.sbi.bayesflow_baseline --all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import bayesflow as bf
import keras
import numpy as np
import torch

from pfn_testing.sbi.plotting import plot_diagnostics, plot_bayesflow_summary
from pfn_testing.sbi.sbibm_utils import (
    AVAILABLE_TASKS,
    evaluate_posterior,
    simulate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    task_name: str = "two_moons"

    # Simulation
    n_train: int = 10_000
    n_val: int = 2_000
    seed: int = 42

    # Flow architecture (None = dimension-aware defaults)
    depth: int | None = None
    subnet_widths: list[int] | None = None

    # Training
    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20

    # Evaluation
    n_posterior_samples: int = 10_000
    n_reference_observations: int = 10

    @property
    def output_dir(self) -> Path:
        suffix = f"bayesflow_n{self.n_train}"
        if self.seed != 42:
            suffix += f"_s{self.seed}"
        return Path("pfn_testing/sbi/outputs") / self.task_name / suffix


def get_flow_defaults(dim_theta: int) -> dict:
    """Dimension-aware flow architecture defaults (matches tabpfn_npe)."""
    if dim_theta <= 5:
        return {"depth": 6, "subnet_widths": [128, 128]}
    else:
        return {"depth": 8, "subnet_widths": [256, 256]}


# ═══════════════════════════════════════════════════════════════════════════════
# BayesFlow approximator
# ═══════════════════════════════════════════════════════════════════════════════

def build_approximator(cfg: Config, dim_theta: int) -> bf.ContinuousApproximator:
    """Build a BayesFlow ContinuousApproximator (NPE with CouplingFlow).

    No summary network: x is fixed-dim and passed directly as inference_conditions.
    Standardization of theta is handled by the adapter.
    """
    defaults = get_flow_defaults(dim_theta)
    depth = cfg.depth if cfg.depth is not None else defaults["depth"]
    widths = cfg.subnet_widths if cfg.subnet_widths is not None else defaults["subnet_widths"]

    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
    )

    inference_network = bf.networks.CouplingFlow(
        depth=depth,
        subnet_kwargs={"widths": widths},
    )

    # standardize='inference_variables' auto-fits mean/std from training data
    approximator = bf.ContinuousApproximator(
        adapter=adapter,
        inference_network=inference_network,
        standardize="inference_variables",
    )

    return approximator


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_approximator(
    approximator: bf.ContinuousApproximator,
    data: dict,
    cfg: Config,
) -> dict:
    """Train the BayesFlow approximator on simulated data.

    Returns a history dict compatible with plot_diagnostics:
        {"train_loss", "val_loss", "best_epoch"}
    """
    train_data = {
        "inference_variables": data["thetas_train"],
        "inference_conditions": data["xs_train"],
    }
    val_data = {
        "inference_variables": data["thetas_val"],
        "inference_conditions": data["xs_val"],
    }

    train_dataset = bf.OfflineDataset(
        data=train_data,
        batch_size=cfg.batch_size,
        adapter=approximator.adapter,
        shuffle=True,
    )
    val_dataset = bf.OfflineDataset(
        data=val_data,
        batch_size=cfg.batch_size,
        adapter=approximator.adapter,
        shuffle=False,
    )

    approximator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        restore_best_weights=True,
        verbose=0,
    )

    with torch.enable_grad():
        keras_history = approximator.fit(
            dataset=train_dataset,
            epochs=cfg.max_epochs,
            validation_data=val_dataset,
            callbacks=[early_stop],
            verbose=0,
        )

    train_losses = keras_history.history["loss"]
    val_losses = keras_history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses))

    print(f"  Trained for {len(train_losses)} epochs, best at {best_epoch}")
    print(f"  Best val loss: {val_losses[best_epoch]:.4f}")

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_epoch": best_epoch,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def make_sample_fn(
    approximator: bf.ContinuousApproximator,
    n_samples: int,
):
    """Return a sample_fn(x_obs_np) -> (n_samples, dim_theta) compatible with evaluate_posterior."""

    def sample_fn(x_obs_np: np.ndarray) -> np.ndarray:
        conditions = {"inference_conditions": x_obs_np.reshape(1, -1)}
        result = approximator.sample(num_samples=n_samples, conditions=conditions)
        # Output shape: (n_obs_batch, n_samples, dim_theta) — take the single observation
        samples = np.asarray(result["inference_variables"])[0]  # (n_samples, dim_theta)
        return samples

    return sample_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Run single task
# ═══════════════════════════════════════════════════════════════════════════════

def run_task(cfg: Config) -> dict:
    """Run the full BayesFlow NPE pipeline on a single sbibm task.

    Returns:
        {"task_name", "dim_theta", "dim_x", "c2st_bayesflow": list[float]}
    """
    out = cfg.output_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"BayesFlow NPE Baseline: {cfg.task_name}")
    print("=" * 60)

    # ── 1. Simulate ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Simulating {cfg.n_train + cfg.n_val} samples...")
    data = simulate(cfg.task_name, cfg.n_train, cfg.n_val, cfg.seed)
    task = data["task"]
    dim_theta, dim_x = data["dim_theta"], data["dim_x"]
    print(f"  Task: {cfg.task_name} (dim_theta={dim_theta}, dim_x={dim_x})")
    print(f"  Train: {data['thetas_train'].shape}, Val: {data['thetas_val'].shape}")

    defaults = get_flow_defaults(dim_theta)
    depth = cfg.depth if cfg.depth is not None else defaults["depth"]
    widths = cfg.subnet_widths if cfg.subnet_widths is not None else defaults["subnet_widths"]
    print(f"  Flow: depth={depth}, widths={widths}")

    # ── 2. Build and train approximator ──────────────────────────────────────
    print("\n[2/4] Building BayesFlow approximator...")
    approximator = build_approximator(cfg, dim_theta)

    print(f"\n[3/4] Training (max_epochs={cfg.max_epochs}, patience={cfg.patience})...")
    history = train_approximator(approximator, data, cfg)

    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating on {cfg.n_reference_observations} sbibm observations...")
    sample_fn = make_sample_fn(approximator, cfg.n_posterior_samples)
    c2st_scores = evaluate_posterior(
        task, sample_fn, cfg.n_reference_observations, cfg.n_posterior_samples,
    )

    # ── Diagnostics for obs 1 ────────────────────────────────────────────────
    x_obs_1 = task.get_observation(num_observation=1).numpy().squeeze(0)
    ref_1 = task.get_reference_posterior_samples(num_observation=1).numpy()
    post_1 = sample_fn(x_obs_1)

    plot_diagnostics(
        post_1, ref_1, history,
        "BayesFlow NPE", str(out / "plots" / "bayesflow_diagnostics.png"),
    )

    # ── Save weights ─────────────────────────────────────────────────────────
    approximator.save_weights(str(out / "weights" / "bayesflow_weights.weights.h5"))

    results = {
        "task_name": cfg.task_name,
        "dim_theta": dim_theta,
        "dim_x": dim_x,
        "c2st_bayesflow": c2st_scores,
    }
    np.savez(
        str(out / "results" / "results.npz"),
        **{k: np.array(v) if isinstance(v, list) else v for k, v in results.items()},
    )

    # ── Print summary ────────────────────────────────────────────────────────
    mean, std = np.mean(c2st_scores), np.std(c2st_scores)
    print(f"\n  BayesFlow C2ST: {mean:.4f} +/- {std:.4f}")
    print(f"  Outputs: {out}/")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results_list: list[dict]) -> None:
    """Print a cross-task summary table."""
    print("\n" + "=" * 60)
    print("BayesFlow Benchmark Summary")
    print("=" * 60)
    print(f"\n{'Task':<28s} {'dim_θ':>5s} {'dim_x':>5s} {'BayesFlow C2ST':>16s}")
    print("-" * 60)
    for r in results_list:
        mean = np.mean(r["c2st_bayesflow"])
        std = np.std(r["c2st_bayesflow"])
        print(
            f"{r['task_name']:<28s} {r['dim_theta']:5d} {r['dim_x']:5d} "
            f"{mean:6.4f}±{std:.4f}"
        )
    print()
    print("(C2ST closer to 0.5 = better)")


def run_benchmark(
    task_names: list[str],
    cfg_overrides: dict | None = None,
) -> list[dict]:
    """Run BayesFlow NPE on multiple tasks and save summary."""
    all_results = []

    for i, task_name in enumerate(task_names, 1):
        print(f"\n{'#' * 60}")
        print(f"# Task {i}/{len(task_names)}: {task_name}")
        print(f"{'#' * 60}\n")

        cfg = Config(task_name=task_name)
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)

        results = run_task(cfg)
        all_results.append(results)

    print_summary(all_results)

    out_dir = Path("pfn_testing/sbi/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_dir / "bayesflow_summary.npz"),
        **{r["task_name"]: np.array(r["c2st_bayesflow"]) for r in all_results},
    )

    if len(all_results) > 1:
        plot_bayesflow_summary(all_results, str(out_dir / "bayesflow_summary.png"))

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BayesFlow NPE Baseline for SBI Benchmarking",
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
        "--max-epochs", type=int, default=None,
        help="Max training epochs (default: 200)",
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
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs

    if args.all:
        run_benchmark(AVAILABLE_TASKS, overrides)
    else:
        run_benchmark([args.task], overrides)


if __name__ == "__main__":
    main()
