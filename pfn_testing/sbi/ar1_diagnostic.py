"""AR1 diagnostic: test flow + training with known sufficient statistics.

Isolates whether posterior inference problems come from:
  (a) the TabPFN embeddings, or
  (b) the normalizing flow / training loop / evaluation setup.

Model:  x_t = alpha + rho * x_{t-1} + eps_t,   eps_t ~ N(0, 1)
Prior:  alpha ~ Uniform(-0.5, 0.5),  rho ~ Uniform(0, 0.99)
Series length: T = 50

Sufficient statistics per series:
  1. sample mean
  2. sample variance
  3. lag-1 autocorrelation

If C2ST ≈ 0.5 → flow/training are fine, problem is in embeddings.
If C2ST >> 0.5 → problem is upstream in simulator or evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pfn_testing.sbi.tabpfn_npe import build_flow, sample_posterior, train_flow
from pfn_testing.sbi.sbibm_utils import compute_c2st
from pfn_testing.sbi.plotting import plot_diagnostics

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ALPHA_MIN, ALPHA_MAX = -0.5, 0.5
RHO_MIN, RHO_MAX = 0.0, 0.99
SIGMA = 1.0
T = 50

N_TRAIN = 5_000
N_VAL = 2_000
N_TEST_OBS = 5
N_POSTERIOR_SAMPLES = 5_000
N_REF_SAMPLES = 10_000


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator
# ═══════════════════════════════════════════════════════════════════════════════


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample (alpha, rho) from the prior. Returns (n, 2)."""
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX, size=n)
    rho = rng.uniform(RHO_MIN, RHO_MAX, size=n)
    return np.column_stack([alpha, rho])


def simulate_ar1(thetas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate AR1 series. thetas: (n, 2), returns: (n, T)."""
    n = thetas.shape[0]
    alpha, rho = thetas[:, 0], thetas[:, 1]

    # Stationary variance: sigma^2 / (1 - rho^2)
    var0 = SIGMA**2 / np.maximum(1.0 - rho**2, 1e-4)
    std0 = np.sqrt(var0)

    x = np.zeros((n, T))
    x[:, 0] = rng.normal(0, 1, size=n) * std0 + alpha / np.maximum(1.0 - rho, 1e-4)

    for t in range(1, T):
        eps = rng.normal(0, SIGMA, size=n)
        x[:, t] = alpha + rho * x[:, t - 1] + eps

    return x


# ═══════════════════════════════════════════════════════════════════════════════
# Sufficient statistics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_sufficient_stats(xs: np.ndarray) -> np.ndarray:
    """Compute (mean, variance, lag-1 autocorrelation) for each series.

    xs: (n, T) → returns (n, 3).
    """
    means = xs.mean(axis=1)
    variances = xs.var(axis=1)

    # Lag-1 autocorrelation per series
    lag1 = np.array(
        [np.corrcoef(x[:-1], x[1:])[0, 1] for x in xs], dtype=np.float64
    )
    # Handle degenerate series (constant → NaN correlation)
    lag1 = np.nan_to_num(lag1, nan=0.0)

    return np.column_stack([means, variances, lag1])


# ═══════════════════════════════════════════════════════════════════════════════
# Grid-based reference posterior  (exact up to grid resolution)
# ═══════════════════════════════════════════════════════════════════════════════


def loglike_ar1(
    x_obs: np.ndarray,
    alpha_grid: np.ndarray,
    rho_grid: np.ndarray,
) -> np.ndarray:
    """Log-likelihood on a 2D (alpha, rho) grid.

    x_obs: (T,), alpha_grid: (n_alpha,), rho_grid: (n_rho,)
    Returns: (n_alpha, n_rho).
    """
    x = x_obs.astype(np.float64)
    alpha = alpha_grid[:, None].astype(np.float64)  # (n_alpha, 1)
    rho = rho_grid[None, :].astype(np.float64)      # (1, n_rho)

    sigma2 = float(SIGMA**2)
    denom = np.maximum(1.0 - rho**2, 1e-4)
    var0 = sigma2 / denom  # (1, n_rho)

    # Stationary mean: alpha / (1 - rho)
    mu0 = alpha / np.maximum(1.0 - rho, 1e-4)  # (n_alpha, n_rho)

    log2pi = math.log(2.0 * math.pi)
    term0 = -0.5 * (log2pi + np.log(var0) + (x[0] - mu0) ** 2 / var0)

    # Transition terms: x_t | x_{t-1} ~ N(alpha + rho * x_{t-1}, sigma^2)
    # residuals shape: (n_alpha, n_rho, T-1)
    residuals = (
        x[1:][None, None, :]
        - alpha[:, :, None]
        - rho[:, :, None] * x[:-1][None, None, :]
    )
    # Note: alpha is (n_alpha, 1) and rho is (1, n_rho), broadcasting gives (n_alpha, n_rho, T-1)
    sq_sum = np.sum(residuals**2, axis=2)  # (n_alpha, n_rho)
    t_minus_1 = len(x) - 1
    term_rest = -0.5 * (t_minus_1 * log2pi + t_minus_1 * math.log(sigma2) + sq_sum / sigma2)

    return term0 + term_rest


def sample_reference_posterior(
    x_obs: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    n_alpha: int = 401,
    n_rho: int = 401,
) -> np.ndarray:
    """Grid-based reference posterior sampling. Returns (n_samples, 2)."""
    alpha_grid = np.linspace(ALPHA_MIN, ALPHA_MAX, n_alpha)
    rho_grid = np.linspace(RHO_MIN, RHO_MAX, n_rho)

    log_lik = loglike_ar1(x_obs, alpha_grid, rho_grid)
    # Flat prior → log-posterior ∝ log-likelihood + const
    log_prior = -math.log(ALPHA_MAX - ALPHA_MIN) - math.log(RHO_MAX - RHO_MIN)
    log_post = log_lik + log_prior

    # Normalize and sample
    flat = log_post.ravel()
    flat = flat - flat.max()
    probs = np.exp(flat)
    probs /= probs.sum()

    flat_idx = rng.choice(probs.size, size=n_samples, replace=True, p=probs)
    alpha_idx, rho_idx = np.unravel_index(flat_idx, log_post.shape)

    return np.column_stack([alpha_grid[alpha_idx], rho_grid[rho_idx]]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal Config for train_flow
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DiagnosticConfig:
    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 30
    grad_clip: float = 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main diagnostic
# ═══════════════════════════════════════════════════════════════════════════════


def run_diagnostic(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    output_dir = Path("pfn_testing/sbi/outputs/ar1_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Simulate training data ────────────────────────────────────────────
    print("[1/5] Simulating AR1 training data...")
    thetas_train = sample_prior(N_TRAIN, rng)
    xs_train = simulate_ar1(thetas_train, rng)
    stats_train = compute_sufficient_stats(xs_train)
    print(f"  Train: thetas {thetas_train.shape}, stats {stats_train.shape}")

    # ── 2. Simulate validation data ──────────────────────────────────────────
    print("[2/5] Simulating AR1 validation data...")
    thetas_val = sample_prior(N_VAL, rng)
    xs_val = simulate_ar1(thetas_val, rng)
    stats_val = compute_sufficient_stats(xs_val)
    print(f"  Val:   thetas {thetas_val.shape}, stats {stats_val.shape}")

    # ── 3. Build and train flow ──────────────────────────────────────────────
    print("[3/5] Training normalizing flow on sufficient statistics...")
    flow = build_flow(
        dim_theta=2,
        dim_context=3,
        n_transforms=5,
        hidden_features=[64, 64],
    )
    cfg = DiagnosticConfig()
    history = train_flow(flow, thetas_train, stats_train, thetas_val, stats_val, cfg)
    print(f"  Best epoch: {history['best_epoch']}")

    # ── 4. Generate test observations + reference posteriors ─────────────────
    print("[4/5] Generating test observations and reference posteriors...")
    rng_test = np.random.default_rng(seed + 999)
    thetas_test = sample_prior(N_TEST_OBS, rng_test)
    xs_test = simulate_ar1(thetas_test, rng_test)

    rng_ref = np.random.default_rng(seed + 5000)

    # ── 5. Evaluate C2ST per test observation ────────────────────────────────
    print(f"[5/5] Evaluating posterior on {N_TEST_OBS} test observations...")
    c2st_scores = []

    for i in range(N_TEST_OBS):
        x_obs = xs_test[i]  # (T,)
        stats_obs = compute_sufficient_stats(x_obs[None, :])[0]  # (3,)

        # Reference posterior via grid
        ref_samples = sample_reference_posterior(x_obs, N_REF_SAMPLES, rng_ref)

        # Flow posterior
        posterior_samples = sample_posterior(
            flow,
            context=stats_obs,
            theta_mean=history["theta_mean"],
            theta_std=history["theta_std"],
            n_samples=N_POSTERIOR_SAMPLES,
        )

        score = compute_c2st(posterior_samples, ref_samples)
        c2st_scores.append(score)
        print(f"  Obs {i + 1}: alpha={thetas_test[i, 0]:.3f}, rho={thetas_test[i, 1]:.3f}, C2ST={score:.4f}")

        # Save diagnostic plot for first observation
        if i == 0:
            plot_diagnostics(
                posterior_samples=posterior_samples,
                reference_samples=ref_samples,
                history=history,
                label="Sufficient Stats Flow",
                output_path=output_dir / "diagnostics_obs1.png",
            )

    mean_c2st = float(np.mean(c2st_scores))
    print(f"\n{'=' * 60}")
    print(f"  Mean C2ST: {mean_c2st:.4f}  (target: ~0.5)")
    print(f"  Per-obs:   {[f'{s:.4f}' for s in c2st_scores]}")
    print(f"{'=' * 60}")

    if mean_c2st < 0.55:
        print("  VERDICT: Flow + training are FINE. Problem is in TabPFN embeddings.")
    elif mean_c2st < 0.65:
        print("  VERDICT: Flow is marginal. Check both embeddings and flow setup.")
    else:
        print("  VERDICT: Flow is BROKEN. Problem is upstream (simulator or evaluation).")

    # Save all results
    np.savez(
        output_dir / "results.npz",
        c2st_scores=np.array(c2st_scores),
        thetas_test=thetas_test,
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"]),
    )
    print(f"\n  Results saved to {output_dir}")


if __name__ == "__main__":
    run_diagnostic()
