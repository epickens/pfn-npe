"""Copula posterior estimators over quantile-probe marginals.

Implements Sklar's decomposition: posterior p(θ|x) = product of marginals
F_d(θ_d|x) joined by a copula c(u_1, ..., u_D | x) on uniformized samples.
Targets joint-structure errors by assigning marginal recovery to a dedicated
quantile probe while the copula models dependence between θ-dimensions.

Two variants:

* ``GaussianCopulaSBI`` — constant Σ estimated from validation Z's. Cheap
  but limited to *linear* cross-dim correlation post-Gaussianization.

* ``NeuralCopulaSBI`` — replaces Σ with a learned conditional NSF on Z
  given encoder(x). Captures non-linear / multi-modal dependence
  structure the Gaussian copula is structurally blind to.

Pipeline (both variants):
    1. Train a per-dim quantile probe on (encoder(x), θ).
    2. Compute predicted quantiles q_τ(x_val) per validation point.
    3. Piecewise-linear CDF F_d(θ_d|x) interpolates the (τ, q_τ) knots.
    4. Uniformize: U_d = F_d(θ_d_val | x_val) ∈ [eps, 1-eps].
    5. Gaussianize: Z_d = Φ⁻¹(U_d).
    6. Fit dependence model on Z (constant Σ for Gaussian, conditional
       NSF for Neural).
    7. Inference: sample Z, U = Φ(Z), invert via F_d⁻¹.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy.stats import norm
from sklearn.model_selection import KFold
from torch.distributions import Distribution

from pfn_testing.sbi.density_estimators import (
    build_flow, get_flow_defaults, sample_posterior, train_flow,
)
from scripts.layer_linear_probe import fit_mean_probe
from scripts.layer_quantile_probe import fit_quantile_probe


DEFAULT_TAUS: tuple = (
    0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975,
)


class GaussianCopulaSBI:
    """Gaussian copula on quantile-probe marginals.

    The model decomposes p(θ|x) as

        p(θ|x) = c(F_1(θ_1|x), ..., F_D(θ_D|x); Σ) × ∏_d f_d(θ_d|x)

    where F_d is a piecewise-linear CDF defined by the predicted quantile
    knots and the copula c is a constant-Σ Gaussian copula on
    Z = Φ⁻¹(F_d(θ_d|x)).
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        seed: int = 42,
        taus: tuple = DEFAULT_TAUS,
        sigma_jitter: float = 1e-6,
        eps: float = 1e-6,
    ) -> None:
        self.prior = prior
        self.seed = seed
        self.taus = np.asarray(taus, dtype=np.float64)
        self.sigma_jitter = sigma_jitter
        self.eps = eps
        # Populated by fit():
        self._quantile_model = None
        self._sigma: np.ndarray | None = None
        self._dim_theta: int | None = None
        # Persistent rng so successive sample() calls advance state.
        self._rng = np.random.default_rng(seed)

    @property
    def sigma(self) -> np.ndarray | None:
        return self._sigma

    @property
    def correlation(self) -> np.ndarray | None:
        if self._sigma is None:
            return None
        d = np.sqrt(np.diag(self._sigma))
        return self._sigma / (d[:, None] * d[None, :])

    def fit(
        self,
        theta_train: np.ndarray,
        theta_val: np.ndarray,
        emb_train: np.ndarray,
        emb_val: np.ndarray,
    ) -> None:
        """Train quantile probe + estimate copula covariance.

        Σ is estimated from validation-set uniformized residuals to
        avoid in-sample bias from the quantile probe's training-set fit.
        """
        self._fit_quantile_probe(theta_train, theta_val, emb_train, emb_val)
        z_val = self._compute_val_z(theta_val, emb_val)
        # Center z's (should already be ~0 mean if marginals are well-fit)
        # then estimate Σ.
        z_centered = z_val - z_val.mean(axis=0, keepdims=True)
        self._sigma = (
            np.cov(z_centered, rowvar=False)
            + self.sigma_jitter * np.eye(self._dim_theta)
        )

    def _fit_quantile_probe(
        self,
        theta_train: np.ndarray,
        theta_val: np.ndarray,
        emb_train: np.ndarray,
        emb_val: np.ndarray,
    ) -> None:
        """Train the per-dim quantile probe on (encoder(x), θ)."""
        mean_best = fit_mean_probe(emb_train, theta_train, emb_val, theta_val)
        self._alpha_mu = mean_best["alpha"]
        q_best = fit_quantile_probe(
            emb_train, theta_train, emb_val, theta_val,
            alpha_mu=self._alpha_mu, taus=tuple(self.taus.tolist()),
        )
        self._quantile_model = q_best["model"]
        self._alpha_q = q_best["alpha"]
        self._dim_theta = int(theta_train.shape[1])

    def _compute_val_z(
        self,
        theta_val: np.ndarray,
        emb_val: np.ndarray,
    ) -> np.ndarray:
        """Compute Z = Φ⁻¹(F_d(θ_d_val | x_val)) using the full-data probe."""
        q_val = self._quantile_model.predict(emb_val)
        u_val = self._piecewise_linear_cdf(theta_val, q_val)
        return norm.ppf(u_val)

    def _piecewise_linear_cdf(
        self,
        theta: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        """Compute U_d = F_d(θ_d|x) per row, F_d piecewise-linear in (τ, q_τ).

        Args:
            theta: (n, D)
            q:     (n, n_tau, D) — predicted τ-quantiles, monotone in τ.

        Returns:
            (n, D) with values in [eps, 1−eps].
        """
        n, D = theta.shape
        n_tau = self.taus.shape[0]
        u = np.empty((n, D), dtype=np.float64)

        for d in range(D):
            t = theta[:, d]
            qs = q[:, :, d]                              # (n, n_tau)
            # idx in [0, n_tau]: number of τ-knots with q_τ ≤ t
            ge = qs <= t[:, None]
            idx = ge.sum(axis=1)

            below = idx == 0
            above = idx == n_tau
            interior = ~(below | above)

            u_d = np.empty(n, dtype=np.float64)
            u_d[below] = self.taus[0]
            u_d[above] = self.taus[-1]

            if interior.any():
                rows = np.flatnonzero(interior)
                idx_int = idx[rows]                       # in [1, n_tau-1]
                q_lo = qs[rows, idx_int - 1]
                q_hi = qs[rows, idx_int]
                t_int = t[rows]
                frac = (t_int - q_lo) / (q_hi - q_lo + 1e-12)
                tau_lo = self.taus[idx_int - 1]
                tau_hi = self.taus[idx_int]
                u_d[rows] = tau_lo + frac * (tau_hi - tau_lo)

            u[:, d] = np.clip(u_d, self.eps, 1.0 - self.eps)

        return u

    def sample(
        self,
        x_obs: np.ndarray,                   # accepted for API symmetry; unused
        emb_obs: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Sample n posterior draws from the fitted copula at x_obs.

        Args:
            x_obs:   (dim_x,) — only kept for API parity with other estimators.
            emb_obs: (dim_emb,) or (1, dim_emb) — encoder embedding of x_obs.
            n_samples: int

        Returns:
            (n_samples, D) ndarray.
        """
        assert self._quantile_model is not None, "fit() first"
        assert self._sigma is not None
        assert self._dim_theta is not None

        if emb_obs.ndim == 1:
            emb_obs = emb_obs.reshape(1, -1)

        # Predicted quantiles for x_obs: q[0] has shape (n_tau, D)
        q_obs = self._quantile_model.predict(emb_obs)[0]   # (n_tau, D)

        # Sample Z ~ N(0, Σ): (n_samples, D)
        z = self._rng.multivariate_normal(
            mean=np.zeros(self._dim_theta),
            cov=self._sigma,
            size=n_samples,
        )

        # U = Φ(Z), clip
        u = np.clip(norm.cdf(z), self.eps, 1.0 - self.eps)

        # Invert per-dim CDF: piecewise-linear interp of (τ, q_obs[:, d]) at u_d
        theta = np.empty_like(z)
        for d in range(self._dim_theta):
            theta[:, d] = np.interp(u[:, d], self.taus, q_obs[:, d])

        return theta


@dataclass
class _CopulaFlowConfig:
    """Minimal config for `train_flow` — only the fields it actually reads."""
    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 5.0


class NeuralCopulaSBI(GaussianCopulaSBI):
    """Quantile-probe marginals joined by a learned conditional NSF on Z.

    Replaces the Gaussian copula's constant Σ with a normalizing flow
    that models p(Z | x) on R^D — captures non-linear cross-dim
    dependence the Gaussian copula misses (e.g. multimodal joint
    structure on two_moons / slcp / gaussian_mixture).

    Training data for the flow:

      * **OOF training Z's** via 5-fold CV of the quantile probe.
        Avoids in-sample bias of probe-fit residuals while keeping a
        large training corpus (~n_train samples).
      * **Val Z's** computed using the full-data probe (held out from
        probe training, so honest by construction). Used as the flow's
        validation set for early stopping.
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        seed: int = 42,
        taus: tuple = DEFAULT_TAUS,
        sigma_jitter: float = 1e-6,
        eps: float = 1e-6,
        n_folds_oof: int = 5,
        flow_n_transforms: Optional[int] = None,
        flow_hidden: Optional[list[int]] = None,
        flow_n_bins: int = 8,
        max_epochs: int = 200,
        lr: float = 5e-4,
        batch_size: int = 256,
        patience: int = 20,
    ) -> None:
        super().__init__(prior=prior, seed=seed, taus=taus,
                         sigma_jitter=sigma_jitter, eps=eps)
        self.n_folds_oof = n_folds_oof
        self.flow_n_transforms = flow_n_transforms
        self.flow_hidden = flow_hidden
        self.flow_n_bins = flow_n_bins
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience

        # Populated by fit():
        self._flow: Optional[torch.nn.Module] = None
        self._flow_history: Optional[dict] = None
        self._oof_z: Optional[np.ndarray] = None  # diagnostic only

    def fit(
        self,
        theta_train: np.ndarray,
        theta_val: np.ndarray,
        emb_train: np.ndarray,
        emb_val: np.ndarray,
    ) -> None:
        """Train quantile probe, OOF-Gaussianize, train conditional NSF on Z."""
        # 1. Full-data quantile probe (sets self._quantile_model, self._dim_theta).
        print("  [neural_copula] fitting full-data quantile probe...")
        self._fit_quantile_probe(theta_train, theta_val, emb_train, emb_val)

        # 2. OOF Z on training set: 5-fold CV of the probe.
        print(f"  [neural_copula] computing OOF Z over {self.n_folds_oof} folds...")
        self._oof_z = self._compute_oof_z(theta_train, emb_train)

        # 3. Val Z (using full-data probe — held out from probe training).
        z_val = self._compute_val_z(theta_val, emb_val)

        # Also compute Σ for diagnostic / fallback (uses val Z).
        z_val_centered = z_val - z_val.mean(axis=0, keepdims=True)
        self._sigma = (
            np.cov(z_val_centered, rowvar=False)
            + self.sigma_jitter * np.eye(self._dim_theta)
        )

        # 4. Build conditional NSF.
        defaults = get_flow_defaults(self._dim_theta)
        n_transforms = self.flow_n_transforms or defaults["n_transforms"]
        hidden = self.flow_hidden or defaults["hidden_features"]
        print(f"  [neural_copula] building NSF (n_transforms={n_transforms}, "
              f"hidden={hidden}, n_bins={self.flow_n_bins})...")
        self._flow = build_flow(
            dim_theta=self._dim_theta,
            dim_context=int(emb_train.shape[1]),
            n_transforms=n_transforms,
            hidden_features=hidden,
            n_bins=self.flow_n_bins,
        )

        # 5. Train flow on (oof_Z, emb_train) with val = (z_val, emb_val).
        cfg = _CopulaFlowConfig(
            lr=self.lr, batch_size=self.batch_size,
            max_epochs=self.max_epochs, patience=self.patience,
        )
        print("  [neural_copula] training NSF on Z residuals...")
        self._flow_history = train_flow(
            self._flow,
            self._oof_z.astype(np.float32),
            emb_train.astype(np.float32, copy=False),
            z_val.astype(np.float32),
            emb_val.astype(np.float32, copy=False),
            cfg,
        )

    def _compute_oof_z(
        self,
        theta_train: np.ndarray,
        emb_train: np.ndarray,
    ) -> np.ndarray:
        """5-fold CV of the quantile probe → out-of-fold Z's on training set.

        Each fold trains the quantile probe at the SAME α as the full-data
        probe (so we don't waste compute on per-fold α sweeps). The
        held-out fold's Z is computed using the fold's predicted quantiles.
        """
        n = theta_train.shape[0]
        kf = KFold(
            n_splits=self.n_folds_oof, shuffle=True, random_state=self.seed,
        )
        oof_z = np.zeros((n, self._dim_theta), dtype=np.float64)

        for fold_idx, (tr_idx, ho_idx) in enumerate(kf.split(emb_train)):
            e_tr = emb_train[tr_idx]
            th_tr = theta_train[tr_idx]
            e_ho = emb_train[ho_idx]
            th_ho = theta_train[ho_idx]

            # Single α (the one the full probe selected) — skip sweep.
            q_best = fit_quantile_probe(
                e_tr, th_tr, e_ho, th_ho,
                alpha_mu=self._alpha_mu,
                alphas=(self._alpha_q,),
                taus=tuple(self.taus.tolist()),
            )
            q_ho = q_best["model"].predict(e_ho)
            u_ho = self._piecewise_linear_cdf(th_ho, q_ho)
            oof_z[ho_idx] = norm.ppf(u_ho)
            print(f"    fold {fold_idx + 1}/{self.n_folds_oof} done "
                  f"(n_held_out={len(ho_idx)})")

        return oof_z

    def sample(
        self,
        x_obs: np.ndarray,
        emb_obs: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Sample n posterior draws via flow(Z|x) → Φ → F_d⁻¹."""
        assert self._quantile_model is not None, "fit() first"
        assert self._flow is not None
        assert self._flow_history is not None

        if emb_obs.ndim == 1:
            emb_obs = emb_obs.reshape(1, -1)

        # Predicted quantiles for x_obs from the full-data probe.
        q_obs = self._quantile_model.predict(emb_obs)[0]   # (n_tau, D)

        # Sample Z from the conditional flow. `sample_posterior` un-standardises
        # internally using the cached theta_mean / theta_std.
        z = sample_posterior(
            self._flow,
            np.asarray(emb_obs[0], dtype=np.float32),
            self._flow_history["theta_mean"],
            self._flow_history["theta_std"],
            n_samples,
        )

        # U = Φ(Z), clip, then invert per-dim CDF via piecewise-linear interp.
        u = np.clip(norm.cdf(z), self.eps, 1.0 - self.eps)
        theta = np.empty_like(z)
        for d in range(self._dim_theta):
            theta[:, d] = np.interp(u[:, d], self.taus, q_obs[:, d])

        return theta


class FullyNeuralCopulaSBI(NeuralCopulaSBI):
    """Per-dim 1D NSF marginals + neural copula on Gaussianized residuals.

    Replaces the linear quantile-probe marginals with **per-dim 1D NSFs**
    so both halves of the Sklar decomposition are non-linear flows. This variant
    tests whether stronger marginal estimators improve the downstream copula.

    Pipeline:
      1. Train D 1D conditional NSFs, one per θ-dim. Each models
         p(θ_d | encoder(x)).
      2. Compute z_d = marginal_flow_d.transform.inv(θ_d_normalized; x).
         Each marginal flow has Gaussian base, so z_d is approximately
         N(0,1) marginally on training data.
      3. Train conditional NSF on (z, encoder(x)) — copula on z-space
         (R^D), capturing cross-dim dependence the marginals don't see.
      4. Inference: sample z ~ copula(·|x_obs), then
         θ_d = marginal_flow_d.transform(z_d; x_obs) (un-standardized).
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        seed: int = 42,
        sigma_jitter: float = 1e-6,
        eps: float = 1e-6,
        # Marginal 1D NSF config
        marginal_n_transforms: int = 3,
        marginal_hidden: Optional[list[int]] = None,
        marginal_n_bins: int = 8,
        marginal_max_epochs: int = 200,
        marginal_lr: float = 5e-4,
        marginal_batch_size: int = 256,
        # Copula NSF config (inherits defaults)
        flow_n_transforms: Optional[int] = None,
        flow_hidden: Optional[list[int]] = None,
        flow_n_bins: int = 8,
        max_epochs: int = 200,
        lr: float = 5e-4,
        batch_size: int = 256,
        patience: int = 20,
    ) -> None:
        # Skip the τ grid — we don't need it for 1D NSF marginals.
        super().__init__(
            prior=prior, seed=seed, taus=DEFAULT_TAUS,
            sigma_jitter=sigma_jitter, eps=eps,
            flow_n_transforms=flow_n_transforms,
            flow_hidden=flow_hidden,
            flow_n_bins=flow_n_bins,
            max_epochs=max_epochs, lr=lr,
            batch_size=batch_size, patience=patience,
        )
        self.marginal_n_transforms = marginal_n_transforms
        self.marginal_hidden = marginal_hidden or [64, 64]
        self.marginal_n_bins = marginal_n_bins
        self.marginal_max_epochs = marginal_max_epochs
        self.marginal_lr = marginal_lr
        self.marginal_batch_size = marginal_batch_size
        # Populated by fit():
        self._marginal_flows: list[torch.nn.Module] = []
        self._marginal_histories: list[dict] = []

    def fit(
        self,
        theta_train: np.ndarray,
        theta_val: np.ndarray,
        emb_train: np.ndarray,
        emb_val: np.ndarray,
    ) -> None:
        """Train D per-dim 1D NSFs, then a conditional NSF on Gaussianized z."""
        self._dim_theta = int(theta_train.shape[1])
        D = self._dim_theta
        emb_train_f = emb_train.astype(np.float32, copy=False)
        emb_val_f = emb_val.astype(np.float32, copy=False)

        # 1. Per-dim 1D NSF marginals.
        self._marginal_flows = []
        self._marginal_histories = []
        for d in range(D):
            print(f"  [fully_neural] training 1D NSF for dim {d+1}/{D}...")
            target_tr = theta_train[:, d:d + 1].astype(np.float32)
            target_va = theta_val[:, d:d + 1].astype(np.float32)
            flow_d = build_flow(
                dim_theta=1, dim_context=int(emb_train_f.shape[1]),
                n_transforms=self.marginal_n_transforms,
                hidden_features=self.marginal_hidden,
                n_bins=self.marginal_n_bins,
            )
            cfg = _CopulaFlowConfig(
                lr=self.marginal_lr, batch_size=self.marginal_batch_size,
                max_epochs=self.marginal_max_epochs, patience=20,
            )
            history = train_flow(
                flow_d, target_tr, emb_train_f, target_va, emb_val_f, cfg,
            )
            self._marginal_flows.append(flow_d)
            self._marginal_histories.append(history)

        # 2. Compute z = marginal_flow.transform.inv(theta_normalized).
        print("  [fully_neural] Gaussianizing train+val via 1D NSF inverse...")
        z_train = self._theta_to_z(theta_train, emb_train_f)
        z_val = self._theta_to_z(theta_val, emb_val_f)

        # Diagnostic Σ for inspection (uses val Z).
        z_val_centered = z_val - z_val.mean(axis=0, keepdims=True)
        self._sigma = (
            np.cov(z_val_centered, rowvar=False)
            + self.sigma_jitter * np.eye(D)
        )

        # 3. Build + train copula NSF on (z, encoder(x)).
        defaults = get_flow_defaults(D)
        n_transforms = self.flow_n_transforms or defaults["n_transforms"]
        hidden = self.flow_hidden or defaults["hidden_features"]
        print(f"  [fully_neural] building copula NSF (n_transforms={n_transforms}, "
              f"hidden={hidden}, n_bins={self.flow_n_bins})...")
        self._flow = build_flow(
            dim_theta=D, dim_context=int(emb_train_f.shape[1]),
            n_transforms=n_transforms, hidden_features=hidden,
            n_bins=self.flow_n_bins,
        )
        cfg = _CopulaFlowConfig(
            lr=self.lr, batch_size=self.batch_size,
            max_epochs=self.max_epochs, patience=self.patience,
        )
        print("  [fully_neural] training copula NSF on z residuals...")
        self._flow_history = train_flow(
            self._flow, z_train.astype(np.float32), emb_train_f,
            z_val.astype(np.float32), emb_val_f, cfg,
        )

    def _theta_to_z(
        self, theta: np.ndarray, emb: np.ndarray,
    ) -> np.ndarray:
        """Map θ → z per-dim via marginal flow inverse on the train-stand. scale.

        Returns: (n, D) Gaussianized residuals.
        """
        n, D = theta.shape
        z = np.zeros((n, D), dtype=np.float64)
        for d, (flow_d, history) in enumerate(zip(
            self._marginal_flows, self._marginal_histories, strict=True,
        )):
            device = next(flow_d.parameters()).device
            t_mean = float(history["theta_mean"][0])
            t_std = float(history["theta_std"][0])
            theta_d_n = (theta[:, d:d + 1] - t_mean) / t_std       # (n, 1)
            theta_d_t = torch.tensor(
                theta_d_n, dtype=torch.float32, device=device,
            )
            emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist = flow_d(emb_t)
                z_d = dist.transform.inv(theta_d_t)               # (n, 1)
            z[:, d] = z_d.cpu().numpy().squeeze(-1)
        return z

    def sample(
        self,
        x_obs: np.ndarray,
        emb_obs: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Sample n posterior draws via copula(z|x) → marginal_flow forward."""
        assert self._flow is not None, "fit() first"
        assert self._flow_history is not None
        assert len(self._marginal_flows) == self._dim_theta

        if emb_obs.ndim == 1:
            emb_obs = emb_obs.reshape(1, -1)
        emb_obs = emb_obs.astype(np.float32, copy=False)

        # 1. Sample z ~ copula(·|emb_obs), un-standardised.
        z = sample_posterior(
            self._flow, np.asarray(emb_obs[0], dtype=np.float32),
            self._flow_history["theta_mean"], self._flow_history["theta_std"],
            n_samples,
        )                                                          # (n_samples, D)

        # 2. Per-dim: θ_d_normalized = marginal_flow_d.transform(z_d; x_obs).
        theta = np.empty_like(z)
        for d, (flow_d, history) in enumerate(zip(
            self._marginal_flows, self._marginal_histories, strict=True,
        )):
            device = next(flow_d.parameters()).device
            t_mean = float(history["theta_mean"][0])
            t_std = float(history["theta_std"][0])
            # Broadcast emb_obs to n_samples
            emb_b = torch.tensor(emb_obs, dtype=torch.float32, device=device)
            emb_b = emb_b.expand(n_samples, -1)
            z_d = torch.tensor(z[:, d:d + 1], dtype=torch.float32, device=device)
            with torch.no_grad():
                dist = flow_d(emb_b)
                theta_d_n = dist.transform(z_d)                   # (n_samples, 1)
            theta_d = theta_d_n.cpu().numpy().squeeze(-1) * t_std + t_mean
            theta[:, d] = theta_d
        return theta
