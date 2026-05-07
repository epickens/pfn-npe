"""Lotka-Volterra with raw (unsubsampled) time series for SBI benchmarking.

4 parameters (alpha, beta, gamma, delta), 402 observation dims
(201 prey + 201 predator time points).
Uses the same ODE and observation model as sbibm, without subsampling.

Reference posteriors via emcee MCMC (4D parameter space).
Cached to disk after first computation.
"""

from __future__ import annotations

import math
from pathlib import Path

import emcee
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.stats import lognorm as scipy_lognorm

# LV prior: LogNormal matching sbibm exactly
_LOG_LOCS = np.array([-0.125, -3.0, -0.125, -3.0])
_LOG_SCALES = np.array([0.5, 0.5, 0.5, 0.5])

_CACHE_DIR = Path(__file__).parent / "data"


def sample_prior(num_samples: int, rng: np.random.Generator | None = None) -> torch.Tensor:
    """Sample theta = (alpha, beta, gamma, delta) from log-normal prior."""
    if rng is None:
        rng = np.random.default_rng()
    log_theta = rng.normal(_LOG_LOCS, _LOG_SCALES, size=(num_samples, 4))
    return torch.tensor(np.exp(log_theta), dtype=torch.float32)


def _solve_lv_ode(alpha: float, beta: float, gamma: float, delta: float) -> np.ndarray | None:
    """Solve LV ODE, return (2, 201) array [prey, predator] or None."""
    u0 = [30.0, 1.0]
    t_eval = np.arange(0, 20.01, 0.1)  # 201 points

    def ode(t, u):
        x, y = u
        return [alpha * x - beta * x * y, -gamma * y + delta * x * y]

    sol = solve_ivp(ode, (0.0, 20.0), u0, t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-8)
    if sol.status != 0 or sol.y.shape[1] < 201:
        return None
    return sol.y[:, :201]  # (2, 201)


def simulate_single(
    alpha: float, beta: float, gamma: float, delta: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Simulate one LV raw observation (402,) or None if ODE fails."""
    sol = _solve_lv_ode(alpha, beta, gamma, delta)
    if sol is None:
        return None
    raw = np.concatenate([sol[0], sol[1]])  # (402,) prey then predator
    raw_clamped = np.clip(raw, 1e-10, 10000)
    obs = rng.lognormal(mean=np.log(raw_clamped), sigma=0.1)
    return obs


def simulate_batch(theta_batch: torch.Tensor, rng: np.random.Generator | None = None) -> torch.Tensor:
    """Simulate batch of LV raw observations."""
    if rng is None:
        rng = np.random.default_rng()
    theta = theta_batch.numpy()
    n = theta.shape[0]
    xs = np.full((n, 402), np.nan)
    for i in range(n):
        result = simulate_single(theta[i, 0], theta[i, 1], theta[i, 2], theta[i, 3], rng)
        if result is not None:
            xs[i] = result
    return torch.tensor(xs, dtype=torch.float32)


def _log_prior(theta: np.ndarray) -> float:
    """Log-prior for LV parameters (log-normal)."""
    if np.any(theta <= 0):
        return -np.inf
    log_theta = np.log(theta)
    return -np.sum(log_theta) - 0.5 * np.sum(((log_theta - _LOG_LOCS) / _LOG_SCALES) ** 2)


def _log_likelihood(theta: np.ndarray, x_obs: np.ndarray) -> float:
    """Log-likelihood: solve ODE + log-normal observation noise."""
    sol = _solve_lv_ode(*theta)
    if sol is None:
        return -np.inf
    raw = np.concatenate([sol[0], sol[1]])
    raw_clamped = np.clip(raw, 1e-10, 10000)
    # Log-normal log-pdf: log(x_obs) ~ N(log(raw_clamped), 0.1^2)
    log_x = np.log(np.maximum(x_obs, 1e-300))
    mu = np.log(raw_clamped)
    return -np.sum(log_x) - 0.5 * np.sum(((log_x - mu) / 0.1) ** 2)


def _log_prob(theta: np.ndarray, x_obs: np.ndarray) -> float:
    """Log-posterior for emcee."""
    lp = _log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = _log_likelihood(theta, x_obs)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def _run_emcee(
    x_obs: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    n_walkers: int = 16,
    n_burn: int = 500,
    n_steps: int = 2000,
) -> torch.Tensor:
    """Run emcee MCMC to get reference posterior samples."""
    ndim = 4

    # Initialize walkers near prior mode with small perturbation in log-space
    log_init = _LOG_LOCS[None, :] + 0.1 * rng.standard_normal((n_walkers, ndim))
    p0 = np.exp(log_init)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, _log_prob, args=(x_obs,))

    # Burn-in
    state = sampler.run_mcmc(p0, n_burn, progress=False)
    sampler.reset()

    # Production
    n_steps_needed = max(n_steps, n_samples // n_walkers + 1)
    sampler.run_mcmc(state, n_steps_needed, progress=False)

    flat = sampler.get_chain(flat=True)
    # Thin to requested number of samples
    if len(flat) > n_samples:
        idx = rng.choice(len(flat), size=n_samples, replace=False)
        flat = flat[idx]

    return torch.tensor(flat, dtype=torch.float32)


class LotkaVolterraRawTask:
    """SBIBM-compatible LV task with full 402-dim raw time series.

    Reference posteriors computed via emcee MCMC and cached to disk.
    """

    dim_theta: int = 4
    dim_x: int = 402

    def __init__(
        self,
        seed: int = 12345,
        n_obs: int = 10,
        n_ref: int = 10_000,
    ) -> None:
        self.n_obs = n_obs
        self.n_ref = n_ref

        rng = np.random.default_rng(seed)

        # Generate reference observations
        theta_star = sample_prior(n_obs, rng)
        self._theta_star = theta_star
        self._x_obs = simulate_batch(theta_star, rng)

        # Resample any failed simulations
        for i in range(n_obs):
            attempts = 0
            while torch.isnan(self._x_obs[i]).any() and attempts < 100:
                theta_star[i] = sample_prior(1, rng).squeeze(0)
                self._x_obs[i] = simulate_batch(theta_star[i : i + 1], rng).squeeze(0)
                attempts += 1

        # Load or compute reference posteriors
        cache_tag = f"lv_raw_s{seed}_n{n_ref}"
        self._reference_posterior_samples = self._load_or_compute_posteriors(
            cache_tag, n_ref, seed,
        )

    def _load_or_compute_posteriors(
        self, cache_tag: str, n_ref: int, seed: int,
    ) -> list[torch.Tensor]:
        cache_path = _CACHE_DIR / f"{cache_tag}.pt"
        if cache_path.exists():
            data = torch.load(cache_path, weights_only=True)
            if len(data) == self.n_obs:
                return data

        print(f"  Computing LV raw reference posteriors via emcee ({self.n_obs} obs)...")
        ref_rng = np.random.default_rng(seed + 10_000)
        samples_list = []
        for i in range(self.n_obs):
            x = self._x_obs[i].numpy().astype(np.float64)
            samples = _run_emcee(x, n_ref, ref_rng)
            samples_list.append(samples)
            print(f"    obs {i+1}/{self.n_obs} done")

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(samples_list, cache_path)
        print(f"  Cached to {cache_path}")
        return samples_list

    def get_prior(self):
        def prior(num_samples: int) -> torch.Tensor:
            return sample_prior(num_samples)
        return prior

    def get_simulator(self):
        def simulator(theta_batch: torch.Tensor) -> torch.Tensor:
            return simulate_batch(theta_batch)
        return simulator

    def get_observation(self, num_observation: int) -> torch.Tensor:
        if not (1 <= num_observation <= self.n_obs):
            raise ValueError(f"num_observation must be in [1, {self.n_obs}]")
        idx = num_observation - 1
        return self._x_obs[idx : idx + 1].clone()

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        if not (1 <= num_observation <= self.n_obs):
            raise ValueError(f"num_observation must be in [1, {self.n_obs}]")
        idx = num_observation - 1
        return self._reference_posterior_samples[idx].clone()
