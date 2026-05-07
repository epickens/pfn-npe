"""SIR model with raw (unsubsampled) time series for SBI benchmarking.

2 parameters (beta, gamma), 161 time points (full ODE trajectory).
Uses the same ODE and observation model as sbibm's SIR task, but returns
all 161 time points instead of subsampling to 10.

Reference posteriors via 2D grid (exact, like ar1_ts_t50).
Cached to disk after first computation (~10 min for 10 obs at 200x200 grid).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.stats import binom as scipy_binom

from pfn_testing.sbi.custom_tasks.grid_posterior import sample_from_nd_grid

# SIR prior: LogNormal matching sbibm exactly
# beta ~ LogNormal(log(0.4), 0.5), gamma ~ LogNormal(log(1/8), 0.2)
_BETA_LOC = math.log(0.4)    # -0.9163
_BETA_SCALE = 0.5
_GAMMA_LOC = math.log(0.125)  # -2.0794
_GAMMA_SCALE = 0.2

# SIR constants
_N_POP = 1_000_000
_N_BINOM = 1000

_CACHE_DIR = Path(__file__).parent / "data"


def sample_prior(num_samples: int, rng: np.random.Generator | None = None) -> torch.Tensor:
    """Sample theta = (beta, gamma) from log-normal prior."""
    if rng is None:
        rng = np.random.default_rng()
    log_beta = rng.normal(_BETA_LOC, _BETA_SCALE, num_samples)
    log_gamma = rng.normal(_GAMMA_LOC, _GAMMA_SCALE, num_samples)
    return torch.tensor(
        np.column_stack([np.exp(log_beta), np.exp(log_gamma)]),
        dtype=torch.float32,
    )


def _solve_sir_ode(beta: float, gamma: float) -> np.ndarray | None:
    """Solve SIR ODE, return infected trajectory (161,) or None."""
    u0 = [_N_POP - 1, 1, 0]
    t_eval = np.arange(0, 161, dtype=float)

    def ode(t, u):
        S, I, R = u
        return [-beta * S * I / _N_POP, beta * S * I / _N_POP - gamma * I, gamma * I]

    sol = solve_ivp(ode, (0.0, 160.0), u0, t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-8)
    if sol.status != 0 or sol.y.shape != (3, 161):
        return None
    return sol.y[1]  # infected trajectory


def simulate_single(beta: float, gamma: float, rng: np.random.Generator) -> np.ndarray | None:
    """Simulate one SIR raw observation (161,) or None if ODE fails."""
    infected = _solve_sir_ode(beta, gamma)
    if infected is None:
        return None
    probs = np.clip(infected / _N_POP, 0, 1)
    return rng.binomial(n=_N_BINOM, p=probs).astype(float)


def simulate_batch(theta_batch: torch.Tensor, rng: np.random.Generator | None = None) -> torch.Tensor:
    """Simulate batch of SIR raw observations."""
    if rng is None:
        rng = np.random.default_rng()
    theta = theta_batch.numpy()
    n = theta.shape[0]
    xs = np.full((n, 161), np.nan)
    for i in range(n):
        result = simulate_single(theta[i, 0], theta[i, 1], rng)
        if result is not None:
            xs[i] = result
    return torch.tensor(xs, dtype=torch.float32)


def _compute_sir_grid_posterior(
    x_obs: np.ndarray,
    grid_size: int,
    n_ref: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Compute reference posterior samples for one observation via 2D grid."""
    # Grid in log-space for better log-normal prior coverage
    log_beta_vals = np.linspace(
        _BETA_LOC - 3 * _BETA_SCALE, _BETA_LOC + 3 * _BETA_SCALE,
        grid_size, dtype=np.float64,
    )
    log_gamma_vals = np.linspace(
        _GAMMA_LOC - 3 * _GAMMA_SCALE, _GAMMA_LOC + 3 * _GAMMA_SCALE,
        grid_size, dtype=np.float64,
    )
    beta_1d = np.exp(log_beta_vals)
    gamma_1d = np.exp(log_gamma_vals)

    obs_int = np.round(x_obs).astype(int)
    ll = np.full((grid_size, grid_size), -1e10, dtype=np.float64)

    for i in range(grid_size):
        for j in range(grid_size):
            infected = _solve_sir_ode(float(beta_1d[i]), float(gamma_1d[j]))
            if infected is None:
                continue
            probs = np.clip(infected / _N_POP, 1e-10, 1 - 1e-10)
            ll[i, j] = scipy_binom.logpmf(obs_int, n=_N_BINOM, p=probs).sum()

    # Add log-normal log-prior
    beta_mesh, gamma_mesh = np.meshgrid(beta_1d, gamma_1d, indexing="ij")
    log_prior = (
        -np.log(beta_mesh) - 0.5 * ((np.log(beta_mesh) - _BETA_LOC) / _BETA_SCALE) ** 2
        - np.log(gamma_mesh) - 0.5 * ((np.log(gamma_mesh) - _GAMMA_LOC) / _GAMMA_SCALE) ** 2
    )
    log_post = ll + log_prior

    return sample_from_nd_grid([beta_1d, gamma_1d], log_post, n_ref, rng)


class SIRRawTask:
    """SBIBM-compatible SIR task with full 161-point time series.

    Reference posteriors are cached to disk after first computation.
    """

    dim_theta: int = 2
    dim_x: int = 161

    def __init__(
        self,
        seed: int = 12345,
        n_obs: int = 10,
        n_ref: int = 10_000,
        grid_size: int = 150,
    ) -> None:
        self.n_obs = n_obs
        self.n_ref = n_ref

        rng = np.random.default_rng(seed)

        # Generate reference observations (deterministic given seed)
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
        cache_tag = f"sir_raw_s{seed}_g{grid_size}_n{n_ref}"
        self._reference_posterior_samples = self._load_or_compute_posteriors(
            cache_tag, grid_size, n_ref, seed,
        )

    def _load_or_compute_posteriors(
        self, cache_tag: str, grid_size: int, n_ref: int, seed: int,
    ) -> list[torch.Tensor]:
        cache_path = _CACHE_DIR / f"{cache_tag}.pt"
        if cache_path.exists():
            data = torch.load(cache_path, weights_only=True)
            if len(data) == self.n_obs:
                return data

        print(f"  Computing SIR raw reference posteriors ({grid_size}x{grid_size} grid, {self.n_obs} obs)...")
        ref_rng = np.random.default_rng(seed + 10_000)
        samples_list = []
        for i in range(self.n_obs):
            x = self._x_obs[i].numpy().astype(np.float64)
            samples = _compute_sir_grid_posterior(x, grid_size, n_ref, ref_rng)
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
