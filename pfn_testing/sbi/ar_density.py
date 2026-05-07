"""Autoregressive density estimator built on TabPFN's per-dim regression.

Implements the chain-rule factorization

    p(θ | x) ≈ ∏ⱼ q(θʲ | θ^{<j}, x)

by per-dimension TabPFN forward passes, mirroring the algorithm of
Vetter et al. 2025 (NPE-PFN, mackelab/npe-pfn). This in-project version
exists so that downstream consumers — autograd-friendly log-prob,
hidden-state probing, AR-derived embeddings → flow — can share the same
TabPFN-internals access patterns we already use in `tabpfn_npe.py` and
`pretrain/encoder.py`.

Phase-1 scope (this file): sklearn-`predict`-based `sample` and
`log_prob`. Query-conditional context filtering is supported (mirrors
NPE-PFN's `standardized_euclidean_filtering`); at n_train ≤
filter_context_size it is a no-op, so all results below the cap match
the un-filtered baseline. No rejection sampling.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from torch import Tensor
from torch.distributions import Distribution

# Match `npe_pfn.npe_pfn` warning filters so our output stays clean too.
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message="`BaseEstimator._validate_data` is deprecated",
)
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message="'force_all_finite' was renamed",
)


def _as_tensor(x: np.ndarray | Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(dtype)
    return torch.as_tensor(x, dtype=dtype)


class TabPFNAR:
    """Autoregressive density estimator over TabPFN regressions.

    The model takes a simulation set (θ, x) and provides `sample(x_obs)`
    and `log_prob(θ, x_obs)` for posterior inference at arbitrary new
    observations. Internally each θ-dim d is modelled by a TabPFN
    regression with features = [x, θ_{<d}] and target = θ_d (chain rule).
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        device: str = "auto",
        seed: int = 42,
        n_estimators: int = 1,
        filter_context_size: int = 10000,
        filter_type: str = "standardized_euclidean",
    ) -> None:
        # n_estimators=1 (default) matches the pattern used by
        # TabPFNEmbedder elsewhere in this project; required for clean
        # hook-based hidden_states extraction. NPE-PFN uses TabPFN's
        # default n_estimators=8, which gives slightly tighter C2ST in
        # exchange for 8× compute and breaks the single-fire hook
        # invariant. The cross-task gap from this mismatch is small
        # (~0.01 joint C2ST in our slcp comparison).
        #
        # filter_context_size: cap on the number of (x, θ) rows passed to
        #     TabPFN as in-context examples per query. At n_train ≤
        #     filter_context_size, the filter is a no-op and the full
        #     simulation set is used (so behavior at our standard
        #     n_train=10000 matches the un-filtered baseline). Above the
        #     cap, filter_type chooses which rows to keep.
        # filter_type:
        #     'standardized_euclidean' (default; NPE-PFN's choice):
        #         z-score x, keep the filter_context_size rows whose x is
        #         closest to x_obs by Euclidean distance. Per-query.
        #     'random': random subset of filter_context_size, fixed by
        #         seed, query-independent.
        #     'none': disable filtering even at oversized n_train; pass
        #         the full set and let TabPFN-v2 subsample internally.
        self.prior = prior
        self.seed = seed
        self.filter_context_size = int(filter_context_size)
        valid_filters = ("standardized_euclidean", "random", "none")
        if filter_type not in valid_filters:
            raise ValueError(
                f"filter_type must be one of {valid_filters}; got {filter_type!r}"
            )
        self.filter_type = filter_type

        kwargs: dict = {"n_estimators": n_estimators, "random_state": seed}
        if device != "auto":
            kwargs["device"] = device
        self._clf = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, **kwargs,
        )

        self._theta_train: Tensor | None = None
        self._x_train: Tensor | None = None

    def _get_context(self, x_obs: Tensor) -> tuple[Tensor, Tensor]:
        """Pick the (θ, x) rows TabPFN should attend to for this query.

        Mirrors NPE-PFN's `support_posterior.standardized_euclidean_filtering`
        (`npe-pfn/npe_pfn/support_posterior.py:357`). At
        n_train ≤ filter_context_size this returns the full stored arrays
        unchanged. Expects a single observation x_obs (shape (1, dim_x)).
        """
        assert self._theta_train is not None and self._x_train is not None
        if self.filter_type == "none" or self.n_train <= self.filter_context_size:
            return self._theta_train, self._x_train

        x_t = _as_tensor(x_obs)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        assert x_t.shape == (1, self.dim_x), (
            f"_get_context expects a single observation (1, {self.dim_x}); "
            f"got {tuple(x_t.shape)}"
        )

        x_train = self._x_train
        if self.filter_type == "standardized_euclidean":
            mu = x_train.mean(dim=0)
            sd = x_train.std(dim=0).clamp_min(1e-8)
            x_s = (x_train - mu) / sd
            obs_s = (x_t - mu) / sd
            dists = torch.cdist(obs_s, x_s).squeeze(0)
            _, idx = torch.topk(
                dists, self.filter_context_size, largest=False,
            )
        else:  # random
            g = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(self.n_train, generator=g)
            idx = perm[:self.filter_context_size]

        return self._theta_train[idx], self._x_train[idx]

    def fit(
        self, theta_train: np.ndarray | Tensor,
        x_train: np.ndarray | Tensor,
    ) -> None:
        """Attach a simulation set. No fitting happens here; per-dim
        TabPFN context fits run lazily inside `sample` and `log_prob`."""
        theta = _as_tensor(theta_train)
        x = _as_tensor(x_train)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        assert theta.ndim == 2 and x.ndim == 2
        assert theta.shape[0] == x.shape[0], "theta and x must align on rows"
        self._theta_train = theta
        self._x_train = x

    @property
    def dim_theta(self) -> int:
        assert self._theta_train is not None, "fit() first"
        return int(self._theta_train.shape[1])

    @property
    def dim_x(self) -> int:
        assert self._x_train is not None, "fit() first"
        return int(self._x_train.shape[1])

    @property
    def n_train(self) -> int:
        assert self._theta_train is not None, "fit() first"
        return int(self._theta_train.shape[0])

    def _validate_x_obs(self, x_obs: np.ndarray | Tensor) -> Tensor:
        x = _as_tensor(x_obs)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.shape == (1, self.dim_x), (
            f"x_obs must be shape (1, {self.dim_x}); got {tuple(x.shape)}"
        )
        return x

    # ─────────────────────────────────────────────────────────────────────
    # Sampling — chain-rule AR via clf.predict()
    # ─────────────────────────────────────────────────────────────────────

    def sample(
        self,
        x_obs: np.ndarray | Tensor,
        n_samples: int,
    ) -> Tensor:
        """Draw n_samples from p(θ | x_obs) via autoregressive sampling.

        Algorithm matches npe_pfn.npe_pfn._sample (lines 111-169) without
        context filtering or rejection sampling.

        Returns:
            Tensor of shape (n_samples, dim_theta).
        """
        assert self._theta_train is not None, "fit() first"
        x = self._validate_x_obs(x_obs)

        theta_ctx, x_ctx = self._get_context(x)
        joint_train = torch.cat([x_ctx, theta_ctx], dim=1)
        dim_x, dim_theta = self.dim_x, self.dim_theta

        # Each chain starts as a copy of x_obs; we append one θ-dim per
        # iteration. Final shape: (n_samples, dim_x + dim_theta).
        samples_batch = x.repeat(n_samples, 1)

        return self._sample_loop(samples_batch, joint_train, dim_x, dim_theta)

    def sample_batched(
        self,
        x_batch: np.ndarray | Tensor,
    ) -> Tensor:
        """Draw one AR sample for each row of x_batch.

        Useful for residual-flow training: each (x_i, θ_i) in the
        training set needs its own θ_AR_i sample. Calling `.sample`
        once per row is wasteful — one D-step AR loop with the entire
        x_batch as the per-row prefix is the same cost as a single
        n_samples=N call.

        Note: query-conditional context filtering does NOT apply here.
        The fit happens once per θ-dim and is shared across all rows of
        x_batch, so per-row filtering would require n_batch × dim_theta
        refits — too expensive for the residual-flow path. At
        n_train > filter_context_size, TabPFN-v2 falls back to its
        internal subsampling.

        Args:
            x_batch: (n, dim_x) tensor or ndarray.

        Returns:
            Tensor of shape (n, dim_theta) — one AR sample per row.
        """
        assert self._theta_train is not None, "fit() first"
        x = _as_tensor(x_batch)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.shape[1] == self.dim_x, (
            f"x_batch must have dim_x={self.dim_x}; got {tuple(x.shape)}"
        )

        joint_train = torch.cat([self._x_train, self._theta_train], dim=1)
        dim_x, dim_theta = self.dim_x, self.dim_theta

        # Each row of samples_batch is its own evolving (x_i, θ_<d_sample_i)
        # vector. Initially just x_i for each row.
        samples_batch = x.clone()
        return self._sample_loop(samples_batch, joint_train, dim_x, dim_theta)

    def _sample_loop(
        self,
        samples_batch: Tensor,
        joint_train: Tensor,
        dim_x: int,
        dim_theta: int,
    ) -> Tensor:
        """Inner AR loop shared by `sample` and `sample_batched`.

        `samples_batch` enters with shape (n, dim_x) — one prefix per row.
        Each iteration appends a sampled θ_d column. Returns the θ part
        only (drops the prefix x columns) at the end.
        """
        for d in range(dim_theta):
            features_end = dim_x + d
            target_idx = dim_x + d

            self._clf.fit(
                joint_train[:, :features_end], joint_train[:, target_idx],
            )
            pred_dist = self._clf.predict(
                samples_batch, output_type="full", quantiles=[],
            )
            param_samples = pred_dist["criterion"].sample(pred_dist["logits"])
            # Bring sample back to the same device as samples_batch.
            param_samples = param_samples.to(samples_batch.device).detach()

            samples_batch = torch.cat(
                [samples_batch, param_samples[:, None]], dim=1,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return samples_batch[:, dim_x:]

    # ─────────────────────────────────────────────────────────────────────
    # Log-prob — chain-rule AR via clf.predict()
    # ─────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────
    # Hidden-state extraction (Phase 2)
    # ─────────────────────────────────────────────────────────────────────

    def _get_transformer_model(self) -> torch.nn.Module:
        """Reach into TabPFN's executor cache for the underlying torch model.

        Mirrors `tabpfn_npe.TabPFNEmbedder._get_transformer_model` (line 269).
        """
        cache = self._clf.executor_.model_caches[0]
        device_key = list(cache._models.keys())[0]
        return cache._models[device_key]

    def hidden_states(
        self,
        theta: np.ndarray | Tensor,
        x: np.ndarray | Tensor,
        layer: int = 11,
        prefix_mode: str = "self",
        return_per_dim: bool = False,
    ) -> np.ndarray | list[np.ndarray]:
        """Per-dim hidden states over the AR forward passes.

        For each θ-dim d ∈ {0..D−1} this fits TabPFN on
        (features=[x_train, θ_train_{<d}], target=θ_train_d), runs a
        forward pass at the test queries (x_test, θ_query_{<d}), and
        captures the encoder output at `layer`. The "test target token"
        (last feature axis) is sliced out — same pattern as
        `TabPFNEmbedder._extract_embeddings` (`tabpfn_npe.py:361-380`).

        Args:
            theta: (n_test, dim_theta) — test parameters; their values
                serve as the AR prefix when prefix_mode='self'.
            x: (1, dim_x) for a single shared x_obs (broadcast across
                all rows of theta), or (n_test, dim_x) for per-row
                conditioning (e.g. for probe training over (θ_val, x_val)
                pairs).
            layer: encoder layer index (0–11).
            prefix_mode: one of:
                - 'self' (default): query prefix = test row's own θ_{<d}.
                  Reproduces NPE-PFN-style test-time conditioning. Means
                  the hidden state directly contains test-θ info — probes
                  built on this concatenation can leak θ.
                - 'mean': query prefix = θ_train.mean(0)[:d]. The same
                  in-distribution constant for every test row, so each
                  hidden state is purely a function of x_test. Probes on
                  this representation isolate the AR encoder's x-encoding
                  contribution from chain-rule prefix exploitation.
                - 'zero': query prefix = 0. OOD; provided for ablations.
            return_per_dim: when True, return a list of per-dim hidden
                states (one ndarray per θ-dim). When False (default),
                return the concatenation across θ-dims.

        Returns:
            If return_per_dim=False: ndarray of shape
            (n_test, dim_theta * emb_dim). For TabPFN's 192-dim
            per-feature encoder, this is (n_test, 192 * dim_theta).
            If return_per_dim=True: list of dim_theta ndarrays each of
            shape (n_test, emb_dim).
        """
        assert self._theta_train is not None, "fit() first"
        theta_t = _as_tensor(theta)
        if theta_t.ndim == 1:
            theta_t = theta_t.unsqueeze(0)
        assert theta_t.shape[1] == self.dim_theta
        n = theta_t.shape[0]

        x_t = _as_tensor(x)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        if x_t.shape[0] == 1 and n > 1:
            x_t = x_t.expand(n, -1)
        assert x_t.shape == (n, self.dim_x), (
            f"x must be (1, {self.dim_x}) or ({n}, {self.dim_x}); "
            f"got {tuple(x_t.shape)}"
        )

        if prefix_mode == "self":
            theta_query = theta_t
        elif prefix_mode == "mean":
            mean_vals = self._theta_train.mean(dim=0).to(theta_t.dtype)
            theta_query = mean_vals.unsqueeze(0).expand(n, -1)
        elif prefix_mode == "zero":
            theta_query = torch.zeros_like(theta_t)
        else:
            raise ValueError(
                f"prefix_mode must be 'self' | 'mean' | 'zero'; got {prefix_mode!r}"
            )

        test_joint = torch.cat([x_t, theta_query], dim=1)

        joint_train = torch.cat([self._x_train, self._theta_train], dim=1)
        dim_x, dim_theta = self.dim_x, self.dim_theta

        per_dim_emb: list[np.ndarray] = []

        for d in range(dim_theta):
            features_end = dim_x + d
            target_idx = dim_x + d
            ctx_features = joint_train[:, :features_end]
            ctx_target = joint_train[:, target_idx]
            test_features = test_joint[:, :features_end]

            self._clf.fit(ctx_features, ctx_target)
            # TabPFN-v2 does not subsample beyond the user-provided context
            # at our standard n_train≤10000; the slice index = ctx rows.
            n_context = ctx_features.shape[0]

            model = self._get_transformer_model()
            encoder = model.transformer_encoder
            captured: dict = {}

            def hook(module, inp, out):
                captured["out"] = out.detach()

            handle = encoder.layers[layer].register_forward_hook(hook)
            try:
                # Trigger a forward pass; predict's output is discarded —
                # we only need the hooked layer activation.
                self._clf.predict(
                    test_features, output_type="full", quantiles=[],
                )
                layer_out = captured["out"]
                # layer_out shape: (1, n_ctx + n_test, n_features + 1, emb_dim)
                test_only = layer_out[:, n_context:, -1]   # target-token pool
                emb_d = test_only.squeeze(0).cpu().numpy()
                if emb_d.ndim == 1:
                    emb_d = emb_d.reshape(1, -1)
            finally:
                handle.remove()

            per_dim_emb.append(emb_d)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if return_per_dim:
            return per_dim_emb
        return np.concatenate(per_dim_emb, axis=1)

    # ─────────────────────────────────────────────────────────────────────
    # Log-prob — chain-rule AR via clf.predict()
    # ─────────────────────────────────────────────────────────────────────

    def log_prob(
        self,
        theta: np.ndarray | Tensor,
        x_obs: np.ndarray | Tensor,
        eps: float = 1e-15,
    ) -> Tensor:
        """log p(θ | x_obs) for a batch of θ values, via chain rule.

        Algorithm matches npe_pfn.npe_pfn._autoregressive_log_prob
        (lines 293-355). Returns Tensor shape (n,).
        """
        assert self._theta_train is not None, "fit() first"
        theta_t = _as_tensor(theta)
        if theta_t.ndim == 1:
            theta_t = theta_t.unsqueeze(0)
        assert theta_t.shape[1] == self.dim_theta, (
            f"theta dim {theta_t.shape[1]} != stored {self.dim_theta}"
        )
        x = self._validate_x_obs(x_obs)

        n = theta_t.shape[0]
        x_batch = x.repeat(n, 1)
        test_joint = torch.cat([x_batch, theta_t], dim=1)

        theta_ctx, x_ctx = self._get_context(x)
        joint_train = torch.cat([x_ctx, theta_ctx], dim=1)
        dim_x, dim_theta = self.dim_x, self.dim_theta

        log_prob = torch.zeros(n)
        for d in range(dim_theta):
            features_end = dim_x + d
            target_idx = dim_x + d

            self._clf.fit(
                joint_train[:, :features_end], joint_train[:, target_idx],
            )
            pred_dist = self._clf.predict(
                test_joint[:, :features_end], output_type="full", quantiles=[],
            )

            dim_log_prob = -pred_dist["criterion"](
                pred_dist["logits"], test_joint[:, target_idx],
            )
            # Replace −inf (out-of-support bin) with log(eps) — same as
            # npe_pfn (line 346).
            dim_log_prob = torch.where(
                dim_log_prob == float("-inf"),
                torch.log(torch.tensor(eps)),
                dim_log_prob,
            )
            log_prob = log_prob + dim_log_prob.detach().to(log_prob.device)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return log_prob
