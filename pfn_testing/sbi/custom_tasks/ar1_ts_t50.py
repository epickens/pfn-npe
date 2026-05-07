"""Custom AR(1) Gaussian time-series task with T=50 for SBI benchmarking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


RHO_MIN = -0.95
RHO_MAX = 0.95
LOG_SIGMA_MIN = math.log(0.05)
LOG_SIGMA_MAX = math.log(2.0)


def sample_prior_torch(
    num_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample theta=(rho, log_sigma) from the task prior."""
    rho = RHO_MIN + (RHO_MAX - RHO_MIN) * torch.rand(
        num_samples, generator=generator, dtype=torch.float32
    )
    log_sigma = LOG_SIGMA_MIN + (LOG_SIGMA_MAX - LOG_SIGMA_MIN) * torch.rand(
        num_samples, generator=generator, dtype=torch.float32
    )
    return torch.stack([rho, log_sigma], dim=1)


def simulate_batch_torch(
    theta_batch: torch.Tensor,
    t: int = 50,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Simulate a batch of AR(1) trajectories from theta=(rho, log_sigma)."""
    if theta_batch.ndim != 2 or theta_batch.shape[1] != 2:
        raise ValueError(
            f"theta_batch must have shape (batch, 2), got {tuple(theta_batch.shape)}"
        )

    theta = theta_batch.to(dtype=torch.float32, device="cpu")
    batch_size = theta.shape[0]
    rho = theta[:, 0]
    sigma = torch.exp(theta[:, 1])

    denom = torch.clamp(1.0 - rho.square(), min=1e-4)
    std0 = sigma / torch.sqrt(denom)

    x = torch.zeros((batch_size, t), dtype=torch.float32)
    x[:, 0] = torch.randn(batch_size, generator=generator, dtype=torch.float32) * std0

    for idx in range(1, t):
        eps = torch.randn(batch_size, generator=generator, dtype=torch.float32) * sigma
        x[:, idx] = rho * x[:, idx - 1] + eps

    return x


def loglike_grid(
    x_obs: np.ndarray,
    rho_grid: np.ndarray,
    log_sigma_grid: np.ndarray,
) -> np.ndarray:
    """Compute vectorized AR(1) log-likelihood on a 2D (rho, log_sigma) grid."""
    x = np.asarray(x_obs, dtype=np.float64)
    rho = np.asarray(rho_grid, dtype=np.float64)  # (n_rho,)
    log_sigma = log_sigma_grid[None, :]  # (1, n_log_sigma)

    sigma2 = np.exp(2.0 * log_sigma)  # (1, n_log_sigma)
    denom = np.maximum(1.0 - np.square(rho), 1e-4)[:, None]  # (n_rho, 1)
    var0 = sigma2 / denom  # (n_rho, n_log_sigma)

    log2pi = math.log(2.0 * math.pi)
    term0 = -0.5 * (log2pi + np.log(var0) + (x[0] ** 2) / var0)

    residuals = x[1:][None, :] - rho[:, None] * x[:-1][None, :]  # (n_rho, t-1)
    sq_sum = np.sum(np.square(residuals), axis=1, keepdims=True)  # (n_rho, 1)
    t_minus_1 = x.shape[0] - 1
    term_rest = -0.5 * (
        t_minus_1 * log2pi + t_minus_1 * np.log(sigma2) + sq_sum / sigma2
    )

    return term0 + term_rest


def sample_from_grid(
    rho_grid: np.ndarray,
    log_sigma_grid: np.ndarray,
    log_post: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sample posterior draws from a discrete normalized 2D grid posterior."""
    flat = log_post.ravel()
    flat = flat - np.max(flat)
    probs = np.exp(flat)
    probs /= probs.sum()

    flat_idx = rng.choice(probs.size, size=n_samples, replace=True, p=probs)
    rho_idx, log_sigma_idx = np.unravel_index(flat_idx, log_post.shape)

    samples = np.column_stack(
        [rho_grid[rho_idx], log_sigma_grid[log_sigma_idx]]
    ).astype(np.float32)
    return torch.from_numpy(samples)


@dataclass(frozen=True)
class _TaskConfig:
    seed: int = 12345
    n_obs: int = 10
    n_ref: int = 10_000
    t: int = 50
    rho_grid_size: int = 401
    log_sigma_grid_size: int = 301


class AR1TimeSeriesT50Task:
    """SBIBM-like AR(1) Gaussian time-series task with dim_x=50 and dim_theta=2."""

    dim_theta: int = 2
    dim_x: int = 50

    def __init__(
        self,
        seed: int = 12345,
        n_obs: int = 10,
        n_ref: int = 10_000,
    ) -> None:
        self.config = _TaskConfig(seed=seed, n_obs=n_obs, n_ref=n_ref)
        self.n_obs = n_obs
        self.n_ref = n_ref

        self._torch_generator = torch.Generator(device="cpu")
        self._torch_generator.manual_seed(seed)

        self._theta_star = sample_prior_torch(
            n_obs, generator=self._torch_generator
        ).to(dtype=torch.float32)
        self._x_obs = simulate_batch_torch(
            self._theta_star, t=self.config.t, generator=self._torch_generator
        ).to(dtype=torch.float32)

        self._rho_grid = np.linspace(
            RHO_MIN, RHO_MAX, self.config.rho_grid_size, dtype=np.float64
        )
        self._log_sigma_grid = np.linspace(
            LOG_SIGMA_MIN,
            LOG_SIGMA_MAX,
            self.config.log_sigma_grid_size,
            dtype=np.float64,
        )
        self._log_prior_const = -math.log(RHO_MAX - RHO_MIN) - math.log(
            LOG_SIGMA_MAX - LOG_SIGMA_MIN
        )

        self._reference_posterior_samples: list[torch.Tensor] = []
        np_rng = np.random.default_rng(seed + 10_000)
        for obs_idx in range(n_obs):
            log_lik = loglike_grid(
                x_obs=self._x_obs[obs_idx].numpy(),
                rho_grid=self._rho_grid,
                log_sigma_grid=self._log_sigma_grid,
            )
            log_post = log_lik + self._log_prior_const
            ref_samples = sample_from_grid(
                rho_grid=self._rho_grid,
                log_sigma_grid=self._log_sigma_grid,
                log_post=log_post,
                n_samples=n_ref,
                rng=np_rng,
            )
            self._reference_posterior_samples.append(ref_samples)

    def get_prior(self):
        """Return prior callable: prior(num_samples) -> (num_samples, dim_theta)."""

        def prior(num_samples: int) -> torch.Tensor:
            return sample_prior_torch(num_samples=num_samples)

        return prior

    def get_simulator(self):
        """Return simulator callable: simulator(theta_batch) -> (batch, dim_x)."""

        def simulator(theta_batch: torch.Tensor) -> torch.Tensor:
            return simulate_batch_torch(theta_batch=theta_batch, t=self.config.t)

        return simulator

    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Return one fixed reference observation with shape (1, dim_x)."""
        if not (1 <= num_observation <= self.n_obs):
            raise ValueError(f"num_observation must be in [1, {self.n_obs}]")
        idx = num_observation - 1
        return self._x_obs[idx : idx + 1].clone()

    def get_true_parameters(self, num_observation: int) -> torch.Tensor:
        """Return ground-truth theta with shape (1, dim_theta)."""
        if not (1 <= num_observation <= self.n_obs):
            raise ValueError(f"num_observation must be in [1, {self.n_obs}]")
        idx = num_observation - 1
        return self._theta_star[idx : idx + 1].clone()

    def get_prior_dist(self) -> torch.distributions.Distribution:
        low = torch.tensor([RHO_MIN, LOG_SIGMA_MIN], dtype=torch.float32)
        high = torch.tensor([RHO_MAX, LOG_SIGMA_MAX], dtype=torch.float32)
        return torch.distributions.Independent(
            torch.distributions.Uniform(low, high), 1
        )

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        """Return cached reference posterior samples with shape (n_ref, dim_theta)."""
        if not (1 <= num_observation <= self.n_obs):
            raise ValueError(f"num_observation must be in [1, {self.n_obs}]")
        idx = num_observation - 1
        return self._reference_posterior_samples[idx].clone()
