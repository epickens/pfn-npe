"""Solar Dynamo (Babcock-Leighton) task for SBI benchmarking.

3 parameters (alpha_min, alpha_range, epsilon_max), 100 time points.
Implements Eqn 2-3 of Charbonneau et al. (2007) and Eqn 13 of Albert et al. (2022).
Matches the SSNL paper specification (Dirmeier & Albert, UAI 2025).

Transition: p_{t+1} = alpha_t * f(p_t) * p_t + epsilon_t
  where alpha_t ~ U(alpha_min, alpha_max), epsilon_t ~ U(0, epsilon_max),
  and f(p) = 0.5 * (1 + erf((p - 0.6)/0.2)) * (1 - erf((p - 1.0)/0.8))

The transition density (sum of two uniforms) is trapezoidal and analytically
tractable, enabling exact grid-based reference posteriors.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import erf as scipy_erf

from pfn_testing.sbi.custom_tasks.grid_posterior import compute_reference_samples


# Prior bounds (matching SSNL)
ALPHA_MIN_LO, ALPHA_MIN_HI = 0.9, 1.4
ALPHA_RANGE_LO, ALPHA_RANGE_HI = 0.05, 0.25
EPS_MAX_LO, EPS_MAX_HI = 0.02, 0.15

# Babcock-Leighton transfer function constants
_B1, _W1, _B2, _W2 = 0.6, 0.2, 1.0, 0.8


def _bl_fn(p: np.ndarray) -> np.ndarray:
    """Babcock-Leighton transfer function f(p)."""
    return 0.5 * (1.0 + scipy_erf((p - _B1) / _W1)) * (1.0 - scipy_erf((p - _B2) / _W2))


def sample_prior(num_samples: int, rng: np.random.Generator | None = None) -> torch.Tensor:
    """Sample theta = (alpha_min, alpha_range, epsilon_max)."""
    if rng is None:
        rng = np.random.default_rng()
    alpha_min = rng.uniform(ALPHA_MIN_LO, ALPHA_MIN_HI, num_samples)
    alpha_range = rng.uniform(ALPHA_RANGE_LO, ALPHA_RANGE_HI, num_samples)
    eps_max = rng.uniform(EPS_MAX_LO, EPS_MAX_HI, num_samples)
    return torch.tensor(
        np.column_stack([alpha_min, alpha_range, eps_max]), dtype=torch.float32
    )


def simulate_batch(
    theta_batch: torch.Tensor,
    n_timepoints: int = 100,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Simulate Solar Dynamo trajectories."""
    if rng is None:
        rng = np.random.default_rng()

    theta = theta_batch.numpy().astype(np.float64)
    batch = theta.shape[0]
    alpha_min = theta[:, 0]
    alpha_max = alpha_min + theta[:, 1]
    eps_max = theta[:, 2]

    # Initial condition: p_0 = 1.0
    p = np.ones(batch, dtype=np.float64)
    trajectory = np.zeros((batch, n_timepoints), dtype=np.float64)

    for t in range(n_timepoints):
        # Sample per-step random variables
        alpha_t = rng.uniform(alpha_min, alpha_max)
        eps_t = rng.uniform(np.zeros(batch), eps_max)

        f = _bl_fn(p)
        p = alpha_t * f * p + eps_t
        trajectory[:, t] = p

    return torch.tensor(trajectory, dtype=torch.float32)


def _log_trapezoidal_pdf(z: np.ndarray, a: np.ndarray, b: np.ndarray,
                         c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Log-PDF of Z = X + Y where X ~ U(a,b), Y ~ U(c,d).

    f_Z(z) = max(0, min(b, z-c) - max(a, z-d)) / ((b-a)*(d-c))
    """
    w1 = b - a  # width of X
    w2 = d - c  # width of Y
    overlap = np.maximum(0.0, np.minimum(b, z - c) - np.maximum(a, z - d))
    denom = w1 * w2
    # Avoid log(0)
    safe_overlap = np.where(overlap > 0, overlap, 1e-300)
    safe_denom = np.where(denom > 0, denom, 1e-300)
    return np.where(overlap > 0, np.log(safe_overlap) - np.log(safe_denom), -1e10)


def solar_dynamo_loglikelihood(
    x_obs: np.ndarray,
    alpha_min: np.ndarray,
    alpha_range: np.ndarray,
    eps_max: np.ndarray,
) -> np.ndarray:
    """Compute log-likelihood on a grid.

    The transition p_{t+1} = alpha_t * f(p_t) * p_t + eps_t
    where alpha_t ~ U(alpha_min, alpha_max), eps_t ~ U(0, eps_max).

    This is a sum of two uniforms with analytically tractable (trapezoidal) density.

    Args:
        x_obs: Observed trajectory, shape (n_timepoints,).
        alpha_min, alpha_range, eps_max: Meshgrid arrays (broadcastable).
    """
    alpha_max = alpha_min + alpha_range

    # Start from p_0 = 1.0 (the initial condition is fixed, not observed)
    p_prev = 1.0
    ll = np.zeros_like(alpha_min, dtype=np.float64)

    for t in range(len(x_obs)):
        p_next = x_obs[t]
        A = _bl_fn(np.asarray(p_prev)) * p_prev  # scalar or array

        # X = A * alpha ~ U(A * alpha_min, A * alpha_max)
        # Y = eps ~ U(0, eps_max)
        # p_next = X + Y
        a = A * alpha_min
        b = A * alpha_max
        c = np.zeros_like(eps_max)
        d = eps_max

        ll += _log_trapezoidal_pdf(p_next, a, b, c, d)
        p_prev = p_next

    return ll


class SolarDynamoTask:
    """SBIBM-compatible Solar Dynamo task."""

    dim_theta: int = 3
    dim_x: int  # set in __init__

    def __init__(
        self,
        seed: int = 12345,
        n_obs: int = 10,
        n_ref: int = 10_000,
        n_timepoints: int = 100,
        grid_size: int = 80,
    ) -> None:
        self.n_obs = n_obs
        self.n_ref = n_ref
        self.n_timepoints = n_timepoints
        self.dim_x = n_timepoints

        rng = np.random.default_rng(seed)

        # Generate reference observations
        theta_star = sample_prior(n_obs, rng)
        self._theta_star = theta_star
        self._x_obs = simulate_batch(theta_star, n_timepoints, rng)

        # Compute grid-based reference posteriors
        grids = [
            np.linspace(ALPHA_MIN_LO, ALPHA_MIN_HI, grid_size, dtype=np.float64),
            np.linspace(ALPHA_RANGE_LO, ALPHA_RANGE_HI, grid_size, dtype=np.float64),
            np.linspace(EPS_MAX_LO, EPS_MAX_HI, grid_size, dtype=np.float64),
        ]

        ref_rng = np.random.default_rng(seed + 10_000)
        self._reference_posterior_samples: list[torch.Tensor] = []
        for i in range(n_obs):
            x = self._x_obs[i].numpy().astype(np.float64)

            def log_lik_fn(am, ar, em, _x=x):
                return solar_dynamo_loglikelihood(_x, am, ar, em)

            samples = compute_reference_samples(grids, log_lik_fn, n_ref, ref_rng)
            self._reference_posterior_samples.append(samples)

    def get_prior(self):
        def prior(num_samples: int) -> torch.Tensor:
            return sample_prior(num_samples)
        return prior

    def get_simulator(self):
        n_tp = self.n_timepoints

        def simulator(theta_batch: torch.Tensor) -> torch.Tensor:
            return simulate_batch(theta_batch, n_tp)
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
        low = torch.tensor(
            [ALPHA_MIN_LO, ALPHA_RANGE_LO, EPS_MAX_LO], dtype=torch.float32
        )
        high = torch.tensor(
            [ALPHA_MIN_HI, ALPHA_RANGE_HI, EPS_MAX_HI], dtype=torch.float32
        )
        return torch.distributions.Independent(
            torch.distributions.Uniform(low, high), 1
        )

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        if not (1 <= num_observation <= self.n_obs):
            raise ValueError(f"num_observation must be in [1, {self.n_obs}]")
        idx = num_observation - 1
        return self._reference_posterior_samples[idx].clone()
