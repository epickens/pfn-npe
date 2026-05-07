"""Shared grid-based reference posterior sampling for low-dimensional tasks.

Exact for 2-3D parameter spaces. Generalizes the approach used in ar1_ts_t50.py
to arbitrary dimensions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def sample_from_nd_grid(
    grids: list[np.ndarray],
    log_posterior: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Draw samples from a discrete N-dimensional grid posterior.

    Args:
        grids: List of 1D arrays, one per parameter dimension.
        log_posterior: N-dimensional array of log-posterior values,
            shape (len(grids[0]), len(grids[1]), ...).
        n_samples: Number of samples to draw.
        rng: NumPy random generator.

    Returns:
        Tensor of shape (n_samples, n_params).
    """
    flat = log_posterior.ravel().astype(np.float64)
    flat -= flat.max()  # log-sum-exp stability
    probs = np.exp(flat)
    probs /= probs.sum()

    flat_idx = rng.choice(probs.size, size=n_samples, replace=True, p=probs)
    nd_idx = np.unravel_index(flat_idx, log_posterior.shape)

    samples = np.column_stack(
        [grids[i][nd_idx[i]] for i in range(len(grids))]
    ).astype(np.float32)
    return torch.from_numpy(samples)


def compute_reference_samples(
    grids: list[np.ndarray],
    log_likelihood_fn: Callable[..., np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Compute reference posterior samples via grid evaluation.

    Assumes a uniform prior over the grid domain.

    Args:
        grids: List of 1D arrays defining the grid axes.
        log_likelihood_fn: Called with meshgrid arrays (one per param dim,
            each broadcastable over the full grid). Must return an array
            of log-likelihood values with the same shape as the grid.
        n_samples: Number of posterior samples to draw.
        rng: NumPy random generator.

    Returns:
        Tensor of shape (n_samples, n_params).
    """
    mesh = np.meshgrid(*grids, indexing="ij")
    log_lik = log_likelihood_fn(*mesh)
    # Uniform prior → log-posterior ∝ log-likelihood (constant prior cancels)
    return sample_from_nd_grid(grids, log_lik, n_samples, rng)
