"""Generic distractor wrapper: adds GMM noise dimensions to any SBIBM task.

The true posterior p(theta | x_real, x_noise) = p(theta | x_real) since the
noise is independent of theta. This lets us reuse the base task's precomputed
reference posteriors for evaluation.

Usage:
    task = DistractorTask("two_moons", noise_dim=92)
    # behaves like any other SBIBM task — works with all pipelines
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


class DistractorTask:
    """Wraps an SBIBM task, appending GMM noise columns to observations."""

    def __init__(
        self,
        base_task_name: str,
        noise_dim: int = 92,
        n_components: int = 5,
        seed: int = 999,
    ) -> None:
        self.base_task_name = base_task_name
        self.noise_dim = noise_dim
        self.seed = seed
        self._n_components = n_components

        # Defer base task loading to avoid circular import with sbibm_utils
        self._base_task = None

        # Set dimensions eagerly using sbibm directly
        import sbibm
        base = sbibm.get_task(base_task_name)
        self.dim_theta = base.dim_parameters
        self.base_dim_x = base.dim_data
        self.dim_x = self.base_dim_x + noise_dim

        # Fit a GMM on synthetic data to define a fixed noise distribution
        rng = np.random.default_rng(seed)
        fit_data = rng.standard_normal((2000, noise_dim))
        for i in range(n_components):
            start = i * (2000 // n_components)
            end = (i + 1) * (2000 // n_components)
            fit_data[start:end] = (
                fit_data[start:end] * rng.uniform(0.5, 3.0, size=noise_dim)
                + rng.uniform(-2, 2, size=noise_dim)
            )

        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=seed,
        )
        self._gmm.fit(fit_data)

        # Fixed column permutation interleaves signal and distractor features.
        self._permutation = rng.permutation(self.dim_x)

    def _get_base_task(self):
        """Lazy-load the base task to avoid circular imports at module load."""
        if self._base_task is None:
            from pfn_testing.sbi.sbibm_utils import get_task
            self._base_task = get_task(self.base_task_name)
        return self._base_task

    def _sample_gmm_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n rows from the fitted GMM using the given RNG."""
        components = rng.choice(
            self._gmm.n_components, size=n, p=self._gmm.weights_,
        )
        noise = np.empty((n, self.noise_dim), dtype=np.float32)
        for k in range(self._gmm.n_components):
            mask = components == k
            count = mask.sum()
            if count > 0:
                noise[mask] = rng.multivariate_normal(
                    self._gmm.means_[k],
                    self._gmm.covariances_[k],
                    size=count,
                ).astype(np.float32)
        return noise

    def _augment(self, x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """Append GMM noise columns and permute."""
        n = x.shape[0]
        rng = np.random.default_rng(seed)
        noise = self._sample_gmm_noise(n, rng)
        noise_t = torch.from_numpy(noise)
        combined = torch.cat([x, noise_t], dim=1)
        return combined[:, self._permutation]

    def get_prior(self):
        """Return prior callable — identical to base task."""
        return self._get_base_task().get_prior()

    def get_prior_dist(self):
        """Return prior distribution — identical to base task."""
        return self._get_base_task().get_prior_dist()

    def get_simulator(self):
        """Return simulator that produces augmented observations.

        ODE base tasks (sir, lotka_volterra) use the Python/scipy simulators
        from sbibm_utils for portability.
        """
        from pfn_testing.sbi.sbibm_utils import ODE_TASKS, _simulate_ode_task

        base_task = self._get_base_task()
        is_ode = self.base_task_name in ODE_TASKS

        def simulator(theta_batch: torch.Tensor) -> torch.Tensor:
            if is_ode:
                thetas_np = theta_batch.numpy()
                xs_np, valid_mask = _simulate_ode_task(
                    self.base_task_name, thetas_np,
                )
                x_real = torch.from_numpy(xs_np.astype(np.float32))
                augmented = self._augment(x_real)
                # Re-insert NaN for failed rows so the caller can filter
                invalid = ~valid_mask
                if invalid.any():
                    augmented[invalid] = float("nan")
                return augmented
            else:
                base_sim = base_task.get_simulator()
                x_real = base_sim(theta_batch)
                return self._augment(x_real)

        return simulator

    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Return base observation augmented with deterministic noise.

        Uses a per-observation seed so the same noise is always appended
        to a given observation number.
        """
        x_real = self._get_base_task().get_observation(
            num_observation=num_observation,
        )
        obs_seed = self.seed + num_observation
        return self._augment(x_real, seed=obs_seed)

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        """Return base task's reference posteriors (noise is theta-independent)."""
        return self._get_base_task().get_reference_posterior_samples(
            num_observation=num_observation,
        )

    def get_true_parameters(self, num_observation: int) -> torch.Tensor:
        """Return base task's true parameters."""
        return self._get_base_task().get_true_parameters(
            num_observation=num_observation,
        )
