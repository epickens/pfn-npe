"""Ornstein-Uhlenbeck process task for SBI benchmarking.

3 parameters (alpha, beta, sigma), configurable time points (default 100).
Analytical Gaussian transitions — no ODE solver needed.

Matches the specification from Dirmeier & Albert (UAI 2025, SSNL paper),
which follows Särkkä & Solin (2019).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pfn_testing.sbi.custom_tasks.grid_posterior import compute_reference_samples


# Prior bounds (matching SSNL)
ALPHA_MIN, ALPHA_MAX = 0.0, 10.0
BETA_MIN, BETA_MAX = 0.0, 5.0
SIGMA_MIN, SIGMA_MAX = 0.0, 2.0

# Small eps to keep beta away from 0 (division in variance formula)
_BETA_EPS = 1e-6


def sample_prior(num_samples: int, rng: np.random.Generator | None = None) -> torch.Tensor:
    """Sample theta = (alpha, beta, sigma) from uniform priors."""
    if rng is None:
        rng = np.random.default_rng()
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX, num_samples)
    beta = rng.uniform(BETA_MIN, BETA_MAX, num_samples)
    sigma = rng.uniform(SIGMA_MIN, SIGMA_MAX, num_samples)
    return torch.tensor(np.column_stack([alpha, beta, sigma]), dtype=torch.float32)


def simulate_batch(
    theta_batch: torch.Tensor,
    n_timepoints: int = 100,
    dt: float = 0.1,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Simulate OU trajectories.

    Transition: x_{t+1} | x_t ~ N(mu_t, var_t)
        mu_t  = alpha + (x_t - alpha) * exp(-beta * dt)
        var_t = sigma^2 * (1 - exp(-2*beta*dt)) / (2*beta)
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = theta_batch.numpy().astype(np.float64)
    batch = theta.shape[0]
    alpha = theta[:, 0]
    beta = np.maximum(theta[:, 1], _BETA_EPS)
    sigma = theta[:, 2]

    # Precompute transition parameters
    exp_neg_bdt = np.exp(-beta * dt)
    var_t = sigma**2 * (1.0 - np.exp(-2.0 * beta * dt)) / (2.0 * beta)
    std_t = np.sqrt(np.maximum(var_t, 1e-12))

    # Stationary distribution for x_0: N(alpha, sigma^2 / (2*beta))
    var_0 = sigma**2 / (2.0 * beta)
    x = np.zeros((batch, n_timepoints), dtype=np.float64)
    x[:, 0] = alpha + rng.standard_normal(batch) * np.sqrt(np.maximum(var_0, 1e-12))

    for t in range(1, n_timepoints):
        mu = alpha + (x[:, t - 1] - alpha) * exp_neg_bdt
        x[:, t] = mu + rng.standard_normal(batch) * std_t

    return torch.tensor(x, dtype=torch.float32)


def ou_loglikelihood(
    x_obs: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """Compute OU log-likelihood on a grid.

    Args:
        x_obs: Observed time series, shape (n_timepoints,).
        alpha, beta, sigma: Meshgrid arrays (broadcastable).
        dt: Time step.

    Returns:
        Log-likelihood array with same shape as the meshgrid.
    """
    beta = np.maximum(beta, _BETA_EPS)
    sigma = np.maximum(sigma, 1e-12)

    exp_neg_bdt = np.exp(-beta * dt)
    var_t = sigma**2 * (1.0 - np.exp(-2.0 * beta * dt)) / (2.0 * beta)
    var_t = np.maximum(var_t, 1e-20)

    # Initial: x_0 ~ N(alpha, sigma^2 / (2*beta))
    var_0 = sigma**2 / (2.0 * beta)
    var_0 = np.maximum(var_0, 1e-20)

    log2pi = math.log(2.0 * math.pi)
    ll = -0.5 * (log2pi + np.log(var_0) + (x_obs[0] - alpha) ** 2 / var_0)

    # Transitions
    for t in range(1, len(x_obs)):
        mu = alpha + (x_obs[t - 1] - alpha) * exp_neg_bdt
        resid_sq = (x_obs[t] - mu) ** 2
        ll = ll - 0.5 * (log2pi + np.log(var_t) + resid_sq / var_t)

    return ll


class OUTask:
    """SBIBM-compatible Ornstein-Uhlenbeck task."""

    dim_theta: int = 3
    dim_x: int  # set in __init__

    def __init__(
        self,
        seed: int = 12345,
        n_obs: int = 10,
        n_ref: int = 10_000,
        n_timepoints: int = 100,
        dt: float = 0.1,
        grid_size: int = 80,
    ) -> None:
        self.n_obs = n_obs
        self.n_ref = n_ref
        self.n_timepoints = n_timepoints
        self.dt = dt
        self.dim_x = n_timepoints

        rng = np.random.default_rng(seed)

        # Generate reference observations
        theta_star = sample_prior(n_obs, rng)
        self._theta_star = theta_star
        self._x_obs = simulate_batch(theta_star, n_timepoints, dt, rng)

        # Compute grid-based reference posteriors
        grids = [
            np.linspace(ALPHA_MIN + 0.01, ALPHA_MAX - 0.01, grid_size, dtype=np.float64),
            np.linspace(BETA_MIN + 0.05, BETA_MAX - 0.01, grid_size, dtype=np.float64),
            np.linspace(SIGMA_MIN + 0.01, SIGMA_MAX - 0.01, grid_size, dtype=np.float64),
        ]

        ref_rng = np.random.default_rng(seed + 10_000)
        self._reference_posterior_samples: list[torch.Tensor] = []
        for i in range(n_obs):
            x = self._x_obs[i].numpy().astype(np.float64)

            def log_lik_fn(a, b, s, _x=x):
                return ou_loglikelihood(_x, a, b, s, dt)

            samples = compute_reference_samples(grids, log_lik_fn, n_ref, ref_rng)
            self._reference_posterior_samples.append(samples)

    def get_prior(self):
        def prior(num_samples: int) -> torch.Tensor:
            return sample_prior(num_samples)
        return prior

    def get_simulator(self):
        n_tp = self.n_timepoints
        _dt = self.dt

        def simulator(theta_batch: torch.Tensor) -> torch.Tensor:
            return simulate_batch(theta_batch, n_tp, _dt)
        return simulator

    def get_observation(self, num_observation: int) -> torch.Tensor:
        if not (1 <= num_observation <= self.n_obs):
            raise ValueError(f"num_observation must be in [1, {self.n_obs}]")
        idx = num_observation - 1
        return self._x_obs[idx : idx + 1].clone()

    def get_true_parameters(self, num_observation: int) -> torch.Tensor:
        if not (1 <= num_observation <= self.n_obs):
            raise ValueError(f"num_observation must be in [1, {self.n_obs}]")
        idx = num_observation - 1
        return self._theta_star[idx : idx + 1].clone()

    def get_prior_dist(self) -> torch.distributions.Distribution:
        low = torch.tensor([ALPHA_MIN, BETA_MIN, SIGMA_MIN], dtype=torch.float32)
        high = torch.tensor([ALPHA_MAX, BETA_MAX, SIGMA_MAX], dtype=torch.float32)
        return torch.distributions.Independent(
            torch.distributions.Uniform(low, high), 1
        )

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        if not (1 <= num_observation <= self.n_obs):
            raise ValueError(f"num_observation must be in [1, {self.n_obs}]")
        idx = num_observation - 1
        return self._reference_posterior_samples[idx].clone()
