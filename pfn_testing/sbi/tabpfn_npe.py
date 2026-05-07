"""
TabPFN Embeddings as Summary Statistics for SBI — Benchmarking

Pipeline:
    Simulator(theta) -> data x -> TabPFN encoder (frozen) -> embeddings s(x)
    -> Normalizing flow q(theta|s(x)) -> Posterior samples

Supports all SBIBM benchmark tasks. Compares TabPFN embeddings against
a raw-features baseline on each task.

Usage:
    uv run python -m pfn_testing.sbi.tabpfn_npe --task two_moons
    uv run python -m pfn_testing.sbi.tabpfn_npe --task slcp --n-train 20000
    uv run python -m pfn_testing.sbi.tabpfn_npe --all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wandb
import zuko
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

from sklearn.decomposition import PCA

from pfn_testing.sbi.load_datasets import load_single_task
from pfn_testing.sbi.plotting import plot_diagnostics, plot_layer_sweep, plot_sbc, plot_summary_table
from pfn_testing.sbi.sbibm_utils import (
    AVAILABLE_TASKS,
    compute_sbc,
    evaluate_posterior,
    get_task,
    simulate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    task_name: str = "two_moons"

    # Simulation
    n_train: int = 10_000
    n_val: int = 2_000
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # TabPFN embedding
    context_size: int = 1_000
    embed_dim: int | None = None  # target dim after reduction (None = full)
    reduction: str = "pca"  # reduction method: "pca", "linear"
    projection_checkpoint: str | None = None  # pretrained LinearProjectionReducer path
    label_strategy: str = "random"  # "random", "constant", "theta_pca", "per_dim", "per_dim_mean"
    layer: int | None = None       # transformer layer to extract (None = last, 0-11)
    model_type: str = "classifier" # "classifier" or "regressor"
    mrl_dim: int | None = None     # MRL: truncate each per-dim embedding to N dims before concat
    pooling: str = "target"        # "target" | "mean" | "first" — feature-axis pooling for hook path

    # Flow architecture (None = use dimension-aware defaults)
    flow_type: str = "nsf"  # "nsf", "naf", "fmpe"
    n_transforms: int | None = None
    hidden_features: list[int] | None = None
    n_bins: int = 8

    # Fine-tuning
    finetune_layers: int = 0       # 0 = frozen, N > 0 = fine-tune last N layers
    lr_tabpfn: float = 1e-5        # LR for TabPFN params
    ft_batch_size: int = 64         # smaller batch size for fine-tuning (more memory per batch)

    # Contrastive-pretrained encoder checkpoint
    encoder_checkpoint: str | None = None  # path to encoder_checkpoint.pt from contrastive_finetune

    # Pre-generated dataset path (skips simulation if set)
    data_path: str | None = None

    # Wandb
    wandb_project: str | None = None

    # Training
    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 5.0

    # Gloeckler decomposition (AR1 only)
    gloeckler_decomp: bool = False

    # Evaluation
    n_posterior_samples: int = 10_000
    n_reference_observations: int = 10
    skip_raw: bool = False  # skip raw-features baseline
    run_sbc: bool = False
    sbc_trials: int = 1000
    sbc_posterior_samples: int = 1000

    @property
    def output_dir(self) -> Path:
        base = Path("pfn_testing/sbi/outputs") / self.task_name
        parts = [
            f"n{self.n_train}",
            self.label_strategy,
        ]
        if self.model_type != "classifier":
            parts.append(self.model_type)
        if self.layer is not None:
            parts.append(f"layer{self.layer}")
        if self.pooling != "target":
            parts.append(self.pooling)
        if self.encoder_checkpoint is not None:
            # Include enough path context to distinguish single vs multi checkpoints.
            # e.g. single: ".../single/task/n10000_contrastive_ft1_s42/encoder_checkpoint.pt"
            #       multi: ".../multi/n10000_ft1_s42/task/n10000_contrastive_ft1_s42/encoder_checkpoint.pt"
            ckpt_path = Path(self.encoder_checkpoint)
            ckpt_label = ckpt_path.parent.name or ckpt_path.stem
            # Detect multi vs single from path
            ckpt_str = str(ckpt_path)
            if "/multi/" in ckpt_str:
                ckpt_label = f"multi_{ckpt_label}"
            parts.append(f"enc_{ckpt_label}")
        if self.mrl_dim is not None:
            parts.append(f"mrl{self.mrl_dim}")
        if self.projection_checkpoint is not None:
            ckpt_name = Path(self.projection_checkpoint).stem  # e.g. "linear_64"
            parts.append(f"pretrained_{ckpt_name}")
        elif self.embed_dim is not None:
            parts.append(f"{self.reduction}_{self.embed_dim}")
        if self.flow_type != "nsf":
            parts.append(self.flow_type)
        if self.context_size != 1_000:
            parts.append(f"ctx{self.context_size}")
        if self.finetune_layers > 0:
            parts.append(f"ft{self.finetune_layers}")
        parts.append(f"s{self.seed}")
        return base / "_".join(parts)


from pfn_testing.sbi.density_estimators import (
    build_flow,
    get_flow_defaults,
    train_flow,
    train_finetuned_flow,
    sample_posterior,
    train_fmpe,
    sample_fmpe_posterior,
    ProjectedFlow,
    FineTunedFlow,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TabPFN Embedding Extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TabPFNEmbedder:
    """Extract frozen TabPFN embeddings, optionally from a specific layer.

    When layer is None (default), uses the built-in get_embeddings API
    which returns final-layer embeddings. When layer is 0-11, uses a
    forward hook to capture intermediate transformer layer outputs.

    Supports both classifier and regressor models. With the regressor,
    labels can be continuous values (e.g., actual theta values for per_dim).

    Label strategies:
        - "random": random binary labels (classifier) or N(0,1) (regressor)
        - "constant": all-zero labels
        - "theta_pca": first PC of theta (binary for classifier, continuous for regressor)
        - "per_dim": one pass per theta dimension, concatenated embeddings.
          Classifier: median-split binary labels. Regressor: raw theta values.
        - "per_dim_mean": like per_dim, but mean-pools across dimensions
          instead of concatenating. Produces a fixed 192-dim embedding.
    """

    def __init__(
        self,
        context_size: int = 1000,
        seed: int = 42,
        batch_size: int = 1000,
        label_strategy: str = "random",
        layer: int | None = None,
        model_type: str = "classifier",
        device: str = "auto",
        mrl_dim: int | None = None,
        pooling: str = "target",
        model_version: str = "v2",
    ):
        self.context_size = context_size
        self.seed = seed
        self.batch_size = batch_size
        self.label_strategy = label_strategy
        self.layer = layer
        self.model_type = model_type
        self.device = device
        self.mrl_dim = mrl_dim  # truncate each per-dim embedding to this many dims (MRL)
        if model_version not in ("v2", "v2.5"):
            raise ValueError(
                f"model_version must be 'v2' or 'v2.5'; got {model_version!r}"
            )
        self.model_version = model_version
        self._mv_enum = (
            ModelVersion.V2 if model_version == "v2" else ModelVersion.V2_5
        )
        if pooling not in ("target", "mean", "first"):
            raise ValueError(f"Unknown pooling: {pooling!r}")
        self.pooling = pooling
        self.clf: TabPFNClassifier | TabPFNRegressor | None = None
        self.emb_dim: int | None = None
        self._n_context: int = 0  # set during fit, needed for hook extraction
        # per_dim state: stored context and label arrays for re-fitting
        self._x_context: np.ndarray | None = None
        self._per_dim_labels: list[np.ndarray] | None = None
        # Encoder checkpoint: state dict to re-apply after each clf.fit()
        self._encoder_state_dict: dict | None = None

    def _create_model(self) -> TabPFNClassifier | TabPFNRegressor:
        """Create a fresh TabPFN model of the configured type."""
        mv = self._mv_enum
        if self.model_type == "regressor":
            if self.device == "auto":
                return TabPFNRegressor.create_default_for_version(
                    mv, n_estimators=1
                )
            return TabPFNRegressor.create_default_for_version(
                mv, n_estimators=1, device=self.device
            )
        if self.device == "auto":
            return TabPFNClassifier.create_default_for_version(
                mv, n_estimators=1
            )
        return TabPFNClassifier.create_default_for_version(
            mv, n_estimators=1, device=self.device
        )

    def _make_labels(
        self,
       
        thetas: np.ndarray | None,
        idx: np.ndarray,
    ) -> np.ndarray:
        """Generate context labels for single-model strategies."""
        n = len(idx)
        if self.label_strategy == "random":
            rng = np.random.default_rng(self.seed)
            if self.model_type == "regressor":
                return rng.standard_normal(n)
            return rng.integers(0, 2, size=n)
        elif self.label_strategy == "constant":
            if self.model_type == "regressor":
                # Regressor short-circuits on constant targets; use tiny noise
                return np.random.default_rng(self.seed).standard_normal(n) * 1e-8
            return np.zeros(n, dtype=int)
        elif self.label_strategy == "theta_pca":
            assert thetas is not None, "theta_pca strategy requires thetas"
            theta_context = thetas[idx]
            if theta_context.shape[1] == 1:
                pc1 = theta_context[:, 0]
            else:
                pc1 = PCA(n_components=1).fit_transform(theta_context)[:, 0]
            if self.model_type == "regressor":
                return pc1.astype(float)
            return (pc1 > np.median(pc1)).astype(int)
        elif self.label_strategy in ("per_dim", "per_dim_mean"):
            raise RuntimeError("per_dim labels are built in fit(), not _make_labels()")
        else:
            raise ValueError(f"Unknown label strategy: {self.label_strategy}")

    def _get_transformer_model(self) -> torch.nn.Module:
        """Access the internal PerFeatureTransformer from a fitted model."""
        cache = self.clf.executor_.model_caches[0]
        device_key = list(cache._models.keys())[0]
        return cache._models[device_key]

    def _apply_encoder_checkpoint(self) -> None:
        """Re-apply encoder checkpoint weights after a clf.fit() call.

        TabPFN recreates the model on each fit(), so we must re-inject
        the fine-tuned weights every time.
        """
        if self._encoder_state_dict is None:
            return
        model = self._get_transformer_model()
        model.load_state_dict(self._encoder_state_dict)

    def load_encoder_checkpoint(self, path: str) -> None:
        """Load contrastive-pretrained encoder weights.

        Must be called after fit(). The weights will be automatically
        re-applied after every subsequent clf.fit() (for per_dim strategy).
        """
        import torch as _torch
        ckpt = _torch.load(path, map_location="cpu", weights_only=False)
        self._encoder_state_dict = ckpt["model_state_dict"]
        self._apply_encoder_checkpoint()
        src_task = ckpt.get("task_name", "unknown")
        src_n = ckpt.get("n_train", "?")
        print(f"  Loaded contrastive encoder: {src_task} (n={src_n})")

    def fit(self, xs: np.ndarray, thetas: np.ndarray | None = None) -> None:
        """Fit TabPFN on a random subset with strategy-determined labels."""
        rng = np.random.default_rng(self.seed)

        self._n_context = min(self.context_size, len(xs))
        idx = rng.choice(len(xs), size=self._n_context, replace=False)
        self._x_context = xs[idx]

        layer_str = f", layer={self.layer}" if self.layer is not None else " (last)"
        model_str = f", {self.model_type}" if self.model_type != "classifier" else ""

        # Create one model (loads weights once)
        self.clf = self._create_model()
        if self.device != "auto":
            self.clf.to(torch.device(self.device))

        if self.label_strategy in ("per_dim", "per_dim_mean"):
            assert thetas is not None, f"{self.label_strategy} strategy requires thetas"
            theta_context = thetas[idx]
            dim_theta = theta_context.shape[1]
            self._per_dim_labels = []
            for d in range(dim_theta):
                col = theta_context[:, d]
                if self.model_type == "regressor":
                    self._per_dim_labels.append(col.astype(float))
                else:
                    self._per_dim_labels.append((col > np.median(col)).astype(int))
            agg = "mean" if self.label_strategy == "per_dim_mean" else "concat"
            mrl_str = f", mrl={self.mrl_dim}" if self.mrl_dim is not None else ""
            print(f"  Label strategy: {self.label_strategy} ({dim_theta} dims, {agg}{model_str}{mrl_str}){layer_str}")
            # Fit with first dim's labels so model is in a fitted state
            self.clf.fit(self._x_context, self._per_dim_labels[0])
        else:
            self._per_dim_labels = None
            y_context = self._make_labels(thetas, idx)
            print(f"  Label strategy: {self.label_strategy}{model_str}{layer_str}")
            self.clf.fit(self._x_context, y_context)

    def _extract_embeddings(self, xs: np.ndarray) -> np.ndarray:
        """Get embeddings from the current clf state for a single batch."""
        if self.layer is None and self.pooling == "target":
            emb = self.clf.get_embeddings(xs, data_source="test")
            # emb shape: (n_estimators, n_samples, emb_dim)
            if emb.shape[0] > 1:
                emb = emb.mean(axis=0)  # Average across estimators
            else:
                emb = emb[0]
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            return emb

        # Hook-based extraction from intermediate (or final, if layer is None
        # with non-target pooling) layer.
        layer_idx = self.layer if self.layer is not None else 11  # final layer
        model = self._get_transformer_model()
        encoder = model.transformer_encoder
        captured = {}

        def hook_fn(module, input, output):
            captured["out"] = output.detach()

        handle = encoder.layers[layer_idx].register_forward_hook(hook_fn)
        try:
            self.clf.get_embeddings(xs, data_source="test")
            layer_out = captured["out"]
            # layer_out shape: (1, n_internal_ctx + n_test, n_features + 1, emb_dim)
            # v2.5 may pad the context internally so n_internal_ctx != self._n_context;
            # slicing from the right is robust to any context-padding scheme.
            test_only = layer_out[:, -xs.shape[0]:]
            if self.pooling == "target":
                test_emb = test_only[:, :, -1]
            elif self.pooling == "mean":
                test_emb = test_only.mean(dim=2)
            elif self.pooling == "first":
                test_emb = test_only[:, :, 0]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling!r}")
            emb = test_emb.squeeze(0).cpu().numpy()
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            return emb
        finally:
            handle.remove()

    def _get_embeddings_single(self, xs: np.ndarray) -> np.ndarray:
        """Get embeddings for a single batch.

        For per_dim: re-fits the single classifier with each dim's labels,
        extracts embeddings, and concatenates. Memory stays constant.
        """
        if self._per_dim_labels is None:
            return self._extract_embeddings(xs)

        # per_dim / per_dim_mean: re-fit and extract for each theta dimension
        parts = []
        for labels in self._per_dim_labels:
            self.clf.fit(self._x_context, labels)
            self._apply_encoder_checkpoint()  # re-inject fine-tuned weights
            emb = self._extract_embeddings(xs)
            if self.mrl_dim is not None:
                emb = emb[:, :self.mrl_dim]
            parts.append(emb)
        if self.label_strategy == "per_dim_mean":
            return np.stack(parts).mean(axis=0)
        return np.concatenate(parts, axis=1)

    def transform(self, xs: np.ndarray) -> np.ndarray:
        """Extract embeddings in batches to avoid OOM.

        Returns:
            embeddings: (n_samples, embedding_dim) array
            For per_dim: embedding_dim = dim_theta * 192
            For per_dim_mean: embedding_dim = 192
        """
        assert self.clf is not None, "Call fit() first"

        n = len(xs)
        if n <= self.batch_size:
            emb = self._get_embeddings_single(xs)
        else:
            chunks = []
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                chunks.append(self._get_embeddings_single(xs[start:end]))
            emb = np.concatenate(chunks, axis=0)

        if self.emb_dim is None:
            self.emb_dim = emb.shape[1]
            print(f"  TabPFN embedding dim: {self.emb_dim}")

        return emb

    def fit_transform(self, xs: np.ndarray, thetas: np.ndarray | None = None) -> np.ndarray:
        self.fit(xs, thetas)
        return self.transform(xs)


class DifferentiableTabPFNEmbedder(torch.nn.Module):
    """Differentiable TabPFN embedder for fine-tuning last N transformer layers.

    Wraps the PerFeatureTransformer for differentiable per_dim embedding
    extraction. Gradients flow through unfrozen transformer layers.
    """

    def __init__(
        self,
        clf: TabPFNRegressor,
        x_context: np.ndarray,
        per_dim_labels: list[np.ndarray],
        n_finetune_layers: int,
    ):
        super().__init__()
        self.n_dims = len(per_dim_labels)

        # Fit clf with each dim's labels to extract preprocessed data
        y_trains: list[torch.Tensor] = []
        cat_ix_list: list[list[int]] = []
        X_train_pp = None
        transform_fn = None

        for d in range(self.n_dims):
            clf.fit(x_context, per_dim_labels[d])
            member = clf.executor_.ensemble_members[0]

            # Store X_train_pp from first dim (same preprocessing across dims)
            if X_train_pp is None:
                X_pp = member.X_train
                if isinstance(X_pp, np.ndarray):
                    X_pp = torch.tensor(X_pp, dtype=torch.float32)
                X_train_pp = X_pp
                transform_fn = member.transform_X_test

            y_pp = member.y_train
            if isinstance(y_pp, np.ndarray):
                y_pp = torch.tensor(y_pp, dtype=torch.float32)
            y_trains.append(y_pp)
            cat_ix_list.append(member.cat_ix)

        # Store as buffers so .to(device) moves them
        self.register_buffer("X_train_pp", X_train_pp)
        for d, y in enumerate(y_trains):
            self.register_buffer(f"y_train_pp_{d}", y)
        self.cat_ix_list = cat_ix_list
        self._transform_fn = transform_fn

        # Access the PerFeatureTransformer model
        model_cache = clf.executor_.model_caches[0]
        device_key = list(model_cache._models.keys())[0]
        self.model = model_cache._models[device_key]

        # Freeze all params, then unfreeze last N layers
        for p in self.model.parameters():
            p.requires_grad = False
        encoder_layers = self.model.transformer_encoder.layers
        n_layers = len(encoder_layers)
        for i in range(max(0, n_layers - n_finetune_layers), n_layers):
            for p in encoder_layers[i].parameters():
                p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"  TabPFN: unfroze last {n_finetune_layers}/{n_layers} layers "
            f"({n_trainable:,}/{n_total:,} trainable params)"
        )

    def preprocess_test(self, xs_test: np.ndarray) -> np.ndarray:
        """Apply CPU preprocessing to test observations."""
        return self._transform_fn(xs_test)

    def forward(self, x_test_pp: torch.Tensor) -> torch.Tensor:
        """Extract concatenated per-dim embeddings differentiably.

        Args:
            x_test_pp: preprocessed test observations, shape (batch_size, n_features)

        Returns:
            Concatenated embeddings, shape (batch_size, n_dims * emb_dim)
        """
        parts = []
        for d in range(self.n_dims):
            # Concat context and test, add batch dim
            X_full = torch.cat([self.X_train_pp, x_test_pp], dim=0).unsqueeze(1)
            # Shape: (n_context + batch_size, 1, n_features)

            y_train = getattr(self, f"y_train_pp_{d}")

            output = self.model(
                X_full,
                y_train,
                only_return_standard_out=False,
                categorical_inds=[self.cat_ix_list[d]],
            )

            # test_embeddings: (n_test, 1, emb_dim) → squeeze batch → (n_test, emb_dim)
            emb_d = output["test_embeddings"]
            if emb_d.dim() == 3:
                emb_d = emb_d.squeeze(1)
            parts.append(emb_d)

        return torch.cat(parts, dim=1)


def extract_ar1_lag_regression_embeddings(
    xs: np.ndarray,
    device: str = "auto",
    series_batch_size: int = 64,
    k: int | None = None,
) -> np.ndarray:
    """Embed AR(1) series via per-series lag regression datasets.

    For each series y of length T, uses either user-provided k or auto k=T//2.
    Then builds:
      X_lag[t] = [y_{t-1}, y_{t-2}, ..., y_{t-k}] for t=k,...,T-1
      y_next = y[k:]
    Then fits a TabPFN regressor on (X_lag, y_next), extracts row embeddings,
    and flattens the full row-embedding matrix to one vector per series.
    """
    if xs.ndim != 2 or xs.shape[1] != 50:
        raise ValueError(f"Expected xs shape (n, 50) for ar1_ts_t50, got {xs.shape}")
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if device == "auto":
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, n_estimators=1)
    else:
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, n_estimators=1, device=device
        )
        model.to(torch.device(device))

    n = xs.shape[0]
    embeddings: list[np.ndarray] = []
    auto_k_logged = False
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for start in range(0, n, series_batch_size):
            end = min(start + series_batch_size, n)
            for i in range(start, end):
                series = xs[i].astype(np.float32, copy=False)
                t_len = series.shape[0]
                k_eff = (t_len // 2) if k is None else k
                if k is None and not auto_k_logged:
                    print(f"  Auto k={k_eff} for T={t_len}")
                    auto_k_logged = True
                if t_len <= k_eff:
                    raise ValueError(f"Series length must be > k ({k_eff}), got {t_len}")
                x_lag = np.stack(
                    [series[k_eff - j : t_len - j] for j in range(1, k_eff + 1)],
                    axis=1,
                ).astype(np.float32, copy=False)
                y_next = series[k_eff:].astype(np.float32, copy=False)
                model.fit(x_lag, y_next)
                emb_rows = model.get_embeddings(x_lag, data_source="test")[0]
                embeddings.append(emb_rows.reshape(-1))
    return np.asarray(embeddings, dtype=np.float32)


def extract_ar1_compositional_embeddings(
    xs: np.ndarray,
    thetas: np.ndarray,
    context_size: int = 1000,
    seed: int = 42,
    device: str = "auto",
) -> tuple[np.ndarray, TabPFNEmbedder]:
    """Compositional transition-based embeddings for AR(1) series.

    Decomposes each series into (x_{t-1}, x_t) transition pairs.
    Builds a cross-simulation context of transitions labelled by (alpha, rho).
    Embeds each transition via TabPFN per_dim regressor, then mean-pools
    across transitions to get one embedding vector per series.

    xs: (n, T), thetas: (n, 2) -> returns (n, emb_dim)
    """
    n, T = xs.shape
    dim_theta = thetas.shape[1]

    # Build transition dataset from ALL training series
    # Each series contributes T-1 transitions
    transitions = []   # each row: [x_{t-1}, x_t]
    trans_thetas = []  # corresponding theta for that series
    for i in range(n):
        for t in range(1, T):
            transitions.append([xs[i, t - 1], xs[i, t]])
            trans_thetas.append(thetas[i])
    transitions = np.array(transitions, dtype=np.float32)    # (n*(T-1), 2)
    trans_thetas = np.array(trans_thetas, dtype=np.float32)  # (n*(T-1), 2)

    # Fit TabPFN embedder on transition pairs with per_dim + regressor
    embedder = TabPFNEmbedder(
        context_size=context_size,
        seed=seed,
        label_strategy="per_dim",
        model_type="regressor",
        device=device,
    )
    embedder.fit(transitions, thetas=trans_thetas)

    # Embed each series by mean-pooling its transition embeddings
    series_embeddings = []
    for i in range(n):
        series_transitions = np.array(
            [[xs[i, t - 1], xs[i, t]] for t in range(1, T)],
            dtype=np.float32,
        )  # (T-1, 2)
        trans_embs = embedder.transform(series_transitions)  # (T-1, emb_dim)
        series_emb = trans_embs.mean(axis=0)                 # (emb_dim,)
        series_embeddings.append(series_emb)

    return np.array(series_embeddings, dtype=np.float32), embedder


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Reduction
# ═══════════════════════════════════════════════════════════════════════════════


class PCAReducer:
    """Reduce embedding dimensionality via PCA."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self._pca = PCA(n_components=n_components)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        result = self._pca.fit_transform(embeddings)
        explained = self._pca.explained_variance_ratio_.cumsum()
        print(
            f"  PCA explained variance: {explained[-1]:.1%} ({self.n_components} components)"
        )
        return result

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self._pca.transform(embeddings)


class LinearProjectionReducer:
    """Learned linear projection via a linear autoencoder.

    Fits a single-layer encoder/decoder to minimize reconstruction error on the
    training embeddings, then uses the encoder as a fixed projection at inference.
    Unlike PCA, the projection is not constrained to be orthogonal.
    """

    def __init__(
        self,
        n_components: int,
        n_epochs: int = 300,
        lr: float = 1e-3,
        batch_size: int = 512,
    ):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self._encoder: torch.nn.Linear | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        # Standardize inputs for stable training
        self._mean = embeddings.mean(axis=0)
        self._std = embeddings.std(axis=0) + 1e-8
        X_np = (embeddings - self._mean) / self._std

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        d_in = X.shape[1]

        encoder = torch.nn.Linear(d_in, self.n_components).to(device)
        decoder = torch.nn.Linear(self.n_components, d_in).to(device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=self.lr
        )

        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        best_loss = float("inf")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                z = encoder(batch)
                recon = decoder(z)
                loss = ((recon - batch) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(X)
            if epoch_loss < best_loss:
                best_loss = epoch_loss

        encoder.eval()
        self._encoder = encoder.cpu()
        print(
            f"  Linear projection: {d_in}d -> {self.n_components}d "
            f"(recon MSE: {best_loss:.4f})"
        )
        with torch.no_grad():
            return self._encoder(torch.tensor(X_np, dtype=torch.float32)).numpy()

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        assert self._encoder is not None, "Must call fit_transform or load first"
        X_np = (embeddings - self._mean) / self._std
        with torch.no_grad():
            return self._encoder(torch.tensor(X_np, dtype=torch.float32)).numpy()

    def save(self, path: str | Path) -> None:
        """Save encoder weights and standardization parameters to a checkpoint."""
        assert self._encoder is not None, "Must call fit_transform before save"
        path = Path(path)
        torch.save(
            {
                "n_components": self.n_components,
                "encoder_state_dict": self._encoder.state_dict(),
                "mean": self._mean,
                "std": self._std,
            },
            path,
        )
        print(f"  Saved linear projection to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LinearProjectionReducer":
        """Load a pretrained reducer from a checkpoint."""
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        reducer = cls(n_components=ckpt["n_components"])
        reducer._mean = ckpt["mean"]
        reducer._std = ckpt["std"]
        d_in = len(reducer._mean)
        reducer._encoder = torch.nn.Linear(d_in, reducer.n_components)
        reducer._encoder.load_state_dict(ckpt["encoder_state_dict"])
        reducer._encoder.eval()
        print(f"  Loaded linear projection from {path} ({d_in}d -> {reducer.n_components}d)")
        return reducer




# ═══════════════════════════════════════════════════════════════════════════════
# Normalizing Flow
# ═══════════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════








# ═══════════════════════════════════════════════════════════════════════════════
# Run single task
# ═══════════════════════════════════════════════════════════════════════════════


def run_task(cfg: Config) -> dict:
    """Run the full TabPFN-NPE pipeline on a single sbibm task.

    Returns:
        {"task_name", "dim_theta", "dim_x", "emb_dim",
         "c2st_tabpfn": list[float], "c2st_raw": list[float]}
    """
    out = cfg.output_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "flows").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    # Wandb
    if cfg.wandb_project:
        wandb.init(
            project=cfg.wandb_project,
            config={k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
            name=f"npe_{cfg.task_name}_n{cfg.n_train}_s{cfg.seed}",
        )

    print("=" * 60)
    print(f"TabPFN-NPE Benchmark: {cfg.task_name}")
    print("=" * 60)

    # ── 1. Load or simulate data ────────────────────────────────────────────
    if cfg.data_path is not None:
        print(f"\n[1/5] Loading dataset from {cfg.data_path}")
        data = load_single_task(cfg.data_path)
        data["task"] = get_task(data["task_name"])
        cfg.task_name = data["task_name"]
        cfg.n_train = len(data["thetas_train"])
        cfg.seed = data["seed"]
    else:
        print(f"\n[1/5] Simulating {cfg.n_train + cfg.n_val} samples...")
        data = simulate(cfg.task_name, cfg.n_train, cfg.n_val, cfg.seed)
    task = data["task"]
    dim_theta, dim_x = data["dim_theta"], data["dim_x"]
    print(f"  Task: {cfg.task_name} (dim_theta={dim_theta}, dim_x={dim_x})")
    print(f"  Train: {data['thetas_train'].shape}, Val: {data['thetas_val'].shape}")

    # Resolve flow architecture
    defaults = get_flow_defaults(dim_theta)
    n_transforms = cfg.n_transforms or defaults["n_transforms"]
    hidden_features = cfg.hidden_features or defaults["hidden_features"]

    if cfg.finetune_layers > 0:
        # ── 2–3. Fine-tuning path: end-to-end TabPFN + flow ─────────────────
        assert cfg.label_strategy == "per_dim", "Fine-tuning requires per_dim label strategy"
        assert cfg.embed_dim is not None, "Fine-tuning requires embed_dim"

        print(f"\n[2/5] Setting up differentiable TabPFN embedder "
              f"(context={cfg.context_size}, finetune_layers={cfg.finetune_layers})...")

        # Prepare context data and per-dim labels
        rng = np.random.default_rng(cfg.seed)
        n_context = min(cfg.context_size, len(data["xs_train"]))
        idx = rng.choice(len(data["xs_train"]), size=n_context, replace=False)
        x_context = data["xs_train"][idx]
        theta_context = data["thetas_train"][idx]
        per_dim_labels = [theta_context[:, d].astype(float) for d in range(dim_theta)]

        clf = TabPFNRegressor.create_default_for_version(ModelVersion.V2, n_estimators=1)
        embedder_module = DifferentiableTabPFNEmbedder(
            clf, x_context, per_dim_labels, cfg.finetune_layers,
        )
        preprocess_fn = embedder_module.preprocess_test

        # Move embedder (buffers) to model's device
        device = next(embedder_module.model.parameters()).device
        embedder_module = embedder_module.to(device)

        # Pre-compute embedding stats for standardization (frozen forward pass)
        with torch.no_grad():
            sample_pp = torch.tensor(
                preprocess_fn(data["xs_train"][:500]),
                dtype=torch.float32, device=device,
            )
            emb_sample = embedder_module(sample_pp)
        emb_mean = emb_sample.mean(dim=0)
        emb_std = emb_sample.std(dim=0) + 1e-8
        d_embedding = emb_sample.shape[1]  # dim_theta * per_dim_emb_size
        emb_dim = cfg.embed_dim

        inner_flow = build_flow(
            dim_theta, cfg.embed_dim, n_transforms, hidden_features, cfg.n_bins, cfg.flow_type,
        )
        flow_tabpfn = FineTunedFlow(
            embedder_module, d_embedding, cfg.embed_dim,
            inner_flow, emb_mean, emb_std,
        )

        # Preprocess all train/val observations
        print("  Preprocessing train/val observations...")
        xs_train_pp = preprocess_fn(data["xs_train"])
        xs_val_pp = preprocess_fn(data["xs_val"])
        print(f"  Train preprocessed: {xs_train_pp.shape}")
        print(f"  Val preprocessed:   {xs_val_pp.shape}")

        print(
            f"\n[3/5] Training fine-tuned flow (embed_dim={cfg.embed_dim}, "
            f"transforms={n_transforms}, hidden={hidden_features}, "
            f"ft_batch={cfg.ft_batch_size}, lr_tabpfn={cfg.lr_tabpfn})..."
        )
        history_tabpfn = train_finetuned_flow(
            flow_tabpfn, data["thetas_train"], xs_train_pp,
            data["thetas_val"], xs_val_pp, cfg,
        )

    else:
        # ── 2. Extract TabPFN embeddings (frozen) ────────────────────────────
        layer_str = f", layer={cfg.layer}" if cfg.layer is not None else ""
        print(f"\n[2/5] Extracting TabPFN embeddings (context={cfg.context_size}{layer_str})...")
        embedder: TabPFNEmbedder | None = None
        ar1_embed_pca: PCA | None = None
        if cfg.task_name == "ar1_ts_t50" and cfg.gloeckler_decomp:
            print("  Embedding mode: compositional transition (Gloeckler decomp)")
            emb_train, embedder = extract_ar1_compositional_embeddings(
                data["xs_train"],
                data["thetas_train"],
                context_size=cfg.context_size,
                seed=cfg.seed,
                device=cfg.device,
            )
            emb_val, _ = extract_ar1_compositional_embeddings(
                data["xs_val"],
                data["thetas_val"],
                context_size=cfg.context_size,
                seed=cfg.seed,
                device=cfg.device,
            )
        elif cfg.task_name == "ar1_ts_t50":
            print("  Embedding mode: cross-simulation per_dim regressor (AR1 fix)")
            embedder = TabPFNEmbedder(
                context_size=cfg.context_size,
                seed=cfg.seed,
                label_strategy="per_dim",
                layer=cfg.layer,
                model_type="regressor",
                device=cfg.device,
            )
            embedder.fit(data["xs_train"], thetas=data["thetas_train"])
            emb_train = embedder.transform(data["xs_train"])
            emb_val = embedder.transform(data["xs_val"])
        else:
            embedder = TabPFNEmbedder(
                context_size=cfg.context_size,
                seed=cfg.seed,
                label_strategy=cfg.label_strategy,
                layer=cfg.layer,
                model_type=cfg.model_type,
                device=cfg.device,
                mrl_dim=cfg.mrl_dim,
            )
            embedder.fit(data["xs_train"], thetas=data["thetas_train"])

            # Load contrastive-pretrained encoder weights if provided
            if cfg.encoder_checkpoint is not None:
                embedder.load_encoder_checkpoint(cfg.encoder_checkpoint)

            emb_train = embedder.transform(data["xs_train"])
            emb_val = embedder.transform(data["xs_val"])
        emb_dim = emb_train.shape[1]
        print(f"  Train embeddings: {emb_train.shape}")
        print(f"  Val embeddings:   {emb_val.shape}")

        # ── 2b. Reduce embedding dimensionality ─────────────────────────────
        reducer = None
        if cfg.projection_checkpoint is not None:
            print(f"\n  Loading pretrained projection: {cfg.projection_checkpoint}")
            reducer = LinearProjectionReducer.load(cfg.projection_checkpoint)
            emb_train = reducer.transform(emb_train)
            emb_val = reducer.transform(emb_val)
            emb_dim = reducer.n_components
            print(f"  Reduced train: {emb_train.shape}")
            print(f"  Reduced val:   {emb_val.shape}")
        elif cfg.embed_dim is not None:
            if cfg.reduction == "pca":
                reducer = PCAReducer(cfg.embed_dim)
            elif cfg.reduction == "linear":
                reducer = LinearProjectionReducer(cfg.embed_dim)
            elif cfg.reduction == "truncate":
                emb_train = emb_train[:, :cfg.embed_dim]
                emb_val = emb_val[:, :cfg.embed_dim]
                emb_dim = cfg.embed_dim
                print(f"\n  Truncating embeddings to first {cfg.embed_dim} dims")
            elif cfg.reduction in ("joint_linear", "joint_linear_pca"):
                pass  # Projection happens inside the flow
            else:
                raise ValueError(f"Unknown reduction: {cfg.reduction}")
            if reducer is not None:
                print(f"\n  Reducing embeddings: {cfg.reduction} -> {cfg.embed_dim}d...")
                emb_train = reducer.fit_transform(emb_train)
                emb_val = reducer.transform(emb_val)
                emb_dim = cfg.embed_dim
                print(f"  Reduced train: {emb_train.shape}")
                print(f"  Reduced val:   {emb_val.shape}")

        # ── 3. Train density estimator on TabPFN embeddings ────────────────────
        posterior_tabpfn = None  # used only for FMPE
        if cfg.flow_type == "fmpe":
            print(f"\n[3/5] Training FMPE on TabPFN embeddings (context={emb_dim})...")
            posterior_tabpfn, history_tabpfn = train_fmpe(
                data["thetas_train"], emb_train,
                data["thetas_val"], emb_val, cfg,
            )
            flow_tabpfn = None
        else:
            print(
                f"\n[3/5] Training {cfg.flow_type.upper()} on TabPFN embeddings (context={emb_dim}, "
                f"transforms={n_transforms}, hidden={hidden_features})..."
            )
            if cfg.reduction in ("joint_linear", "joint_linear_pca") and cfg.embed_dim is not None:
                inner_flow = build_flow(
                    dim_theta, cfg.embed_dim, n_transforms, hidden_features, cfg.n_bins, cfg.flow_type
                )
                emb_mean = torch.tensor(emb_train.mean(axis=0), dtype=torch.float32)
                emb_std = torch.tensor(emb_train.std(axis=0) + 1e-8, dtype=torch.float32)
                pca_components = None
                if cfg.reduction == "joint_linear_pca":
                    standardized = (emb_train - emb_train.mean(axis=0)) / (emb_train.std(axis=0) + 1e-8)
                    pca = PCA(n_components=cfg.embed_dim)
                    pca.fit(standardized)
                    pca_components = torch.tensor(pca.components_, dtype=torch.float32)
                flow_tabpfn = ProjectedFlow(emb_dim, cfg.embed_dim, inner_flow, emb_mean, emb_std, pca_components)
            else:
                flow_tabpfn = build_flow(
                    dim_theta, emb_dim, n_transforms, hidden_features, cfg.n_bins, cfg.flow_type
                )
            history_tabpfn = train_flow(
                flow_tabpfn,
                data["thetas_train"],
                emb_train,
                data["thetas_val"],
                emb_val,
                cfg,
            )

    # ── 4. Train density estimator on raw features (baseline) ────────────────
    flow_raw = None
    posterior_raw = None  # used only for FMPE
    history_raw = None
    if not cfg.skip_raw:
        if cfg.flow_type == "fmpe":
            print(f"\n[4/5] Training FMPE on raw features (baseline, context={dim_x})...")
            posterior_raw, history_raw = train_fmpe(
                data["thetas_train"], data["xs_train"],
                data["thetas_val"], data["xs_val"], cfg,
            )
        else:
            print(f"\n[4/5] Training {cfg.flow_type.upper()} on raw features (baseline, context={dim_x})...")
            flow_raw = build_flow(dim_theta, dim_x, n_transforms, hidden_features, cfg.n_bins, cfg.flow_type)
            history_raw = train_flow(
                flow_raw,
                data["thetas_train"],
                data["xs_train"],
                data["thetas_val"],
                data["xs_val"],
                cfg,
            )
    else:
        print(f"\n[4/5] Skipping raw-features baseline")

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluating on {cfg.n_reference_observations} sbibm observations...")

    def _sample_from_estimator(estimator, posterior_obj, context, history):
        """Sample from either a zuko flow or FMPE posterior."""
        if cfg.flow_type == "fmpe":
            return sample_fmpe_posterior(
                posterior_obj, context,
                history["theta_mean"], history["theta_std"],
                cfg.n_posterior_samples,
            )
        else:
            return sample_posterior(
                estimator, context,
                history["theta_mean"], history["theta_std"],
                cfg.n_posterior_samples,
            )

    if cfg.finetune_layers > 0:
        def tabpfn_sample_fn(x_obs: np.ndarray) -> np.ndarray:
            x_pp = preprocess_fn(x_obs.reshape(1, -1))
            return _sample_from_estimator(flow_tabpfn, posterior_tabpfn, x_pp[0], history_tabpfn)
    else:
        def tabpfn_sample_fn(x_obs: np.ndarray) -> np.ndarray:
            if cfg.task_name == "ar1_ts_t50" and cfg.gloeckler_decomp:
                series = x_obs.reshape(-1)
                T = len(series)
                series_transitions = np.array(
                    [[series[t - 1], series[t]] for t in range(1, T)],
                    dtype=np.float32,
                )
                trans_embs = embedder.transform(series_transitions)
                obs_emb = trans_embs.mean(axis=0, keepdims=True)
            else:
                assert embedder is not None
                obs_emb = embedder.transform(x_obs.reshape(1, -1))
            if reducer is not None:
                obs_emb = reducer.transform(obs_emb)
            elif cfg.reduction == "truncate" and cfg.embed_dim is not None:
                obs_emb = obs_emb[:, :cfg.embed_dim]
            return _sample_from_estimator(flow_tabpfn, posterior_tabpfn, obs_emb[0], history_tabpfn)

    raw_sample_fn = None
    if flow_raw is not None or posterior_raw is not None:
        def raw_sample_fn(x_obs: np.ndarray) -> np.ndarray:
            return _sample_from_estimator(flow_raw, posterior_raw, x_obs, history_raw)

    print("  TabPFN embeddings:")
    c2st_tabpfn = evaluate_posterior(
        task,
        tabpfn_sample_fn,
        cfg.n_reference_observations,
        cfg.n_posterior_samples,
    )
    c2st_raw = []
    if raw_sample_fn is not None:
        print("  Raw features:")
        c2st_raw = evaluate_posterior(
            task,
            raw_sample_fn,
            cfg.n_reference_observations,
            cfg.n_posterior_samples,
        )

    # ── Diagnostics for obs 1 ────────────────────────────────────────────────
    x_obs_1 = task.get_observation(num_observation=1).numpy().squeeze(0)
    ref_1 = task.get_reference_posterior_samples(num_observation=1).numpy()

    post_tabpfn_1 = tabpfn_sample_fn(x_obs_1)

    plot_diagnostics(
        post_tabpfn_1,
        ref_1,
        history_tabpfn,
        "TabPFN emb",
        str(out / "plots" / "tabpfn_emb_diagnostics.png"),
    )
    if raw_sample_fn is not None:
        post_raw_1 = raw_sample_fn(x_obs_1)
        plot_diagnostics(
            post_raw_1,
            ref_1,
            history_raw,
            "Raw features",
            str(out / "plots" / "raw_features_diagnostics.png"),
        )

    # ── SBC evaluation (optional) ──────────────────────────────────────────
    sbc_tabpfn = None
    sbc_raw = None
    if cfg.run_sbc:
        print(f"\n[6/6] Running SBC ({cfg.sbc_trials} trials, "
              f"{cfg.sbc_posterior_samples} posterior samples each)...")

        print("  TabPFN embeddings:")
        sbc_tabpfn = compute_sbc(
            cfg.task_name, tabpfn_sample_fn,
            n_trials=cfg.sbc_trials,
            n_posterior_samples=cfg.sbc_posterior_samples,
        )
        plot_sbc(sbc_tabpfn, "TabPFN emb", str(out / "plots" / "sbc_tabpfn.png"))

        if raw_sample_fn is not None:
            print("  Raw features:")
            sbc_raw = compute_sbc(
                cfg.task_name, raw_sample_fn,
                n_trials=cfg.sbc_trials,
                n_posterior_samples=cfg.sbc_posterior_samples,
                seed=456,
            )
            plot_sbc(sbc_raw, "Raw features", str(out / "plots" / "sbc_raw.png"))

    # ── Save ─────────────────────────────────────────────────────────────────
    if flow_tabpfn is not None:
        torch.save(flow_tabpfn.state_dict(), str(out / "flows" / "flow_tabpfn.pt"))
    if flow_raw is not None:
        torch.save(flow_raw.state_dict(), str(out / "flows" / "flow_raw.pt"))

    results = {
        "task_name": cfg.task_name,
        "dim_theta": dim_theta,
        "dim_x": dim_x,
        "emb_dim": emb_dim,
        "c2st_tabpfn": c2st_tabpfn,
    }
    if c2st_raw:
        results["c2st_raw"] = c2st_raw
    if sbc_tabpfn is not None:
        results["sbc_ks_tabpfn"] = sbc_tabpfn["ks_stats"]
        results["sbc_mean_ks_tabpfn"] = sbc_tabpfn["mean_ks"]
    if sbc_raw is not None:
        results["sbc_ks_raw"] = sbc_raw["ks_stats"]
        results["sbc_mean_ks_raw"] = sbc_raw["mean_ks"]

    np.savez(
        str(out / "results" / "results.npz"),
        **{k: np.array(v) if isinstance(v, list) else v for k, v in results.items()},
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n  TabPFN C2ST: {np.mean(c2st_tabpfn):.4f} +/- {np.std(c2st_tabpfn):.4f}")
    if c2st_raw:
        print(f"  Raw    C2ST: {np.mean(c2st_raw):.4f} +/- {np.std(c2st_raw):.4f}")
    if sbc_tabpfn is not None:
        print(f"  TabPFN SBC mean KS: {sbc_tabpfn['mean_ks']:.4f}")
    if sbc_raw is not None:
        print(f"  Raw    SBC mean KS: {sbc_raw['mean_ks']:.4f}")
    print(f"  Outputs: {out}/")

    if wandb.run:
        log_dict = {
            "c2st_tabpfn_mean": float(np.mean(c2st_tabpfn)),
            "c2st_tabpfn_std": float(np.std(c2st_tabpfn)),
        }
        if c2st_raw:
            log_dict["c2st_raw_mean"] = float(np.mean(c2st_raw))
            log_dict["c2st_raw_std"] = float(np.std(c2st_raw))
        wandb.log(log_dict)
        wandb.finish()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════


def print_summary(results_list: list[dict]) -> None:
    """Print a cross-task comparison table."""
    print("\n" + "=" * 72)
    print("Benchmark Summary")
    print("=" * 72)
    print(
        f"\n{'Task':<28s} {'dim_θ':>5s} {'dim_x':>5s} {'TabPFN C2ST':>14s} {'Raw C2ST':>14s}"
    )
    print("-" * 72)
    for r in results_list:
        t_mean, t_std = np.mean(r["c2st_tabpfn"]), np.std(r["c2st_tabpfn"])
        line = f"{r['task_name']:<28s} {r['dim_theta']:5d} {r['dim_x']:5d} {t_mean:6.4f}±{t_std:.4f}"
        if "c2st_raw" in r:
            r_mean, r_std = np.mean(r["c2st_raw"]), np.std(r["c2st_raw"])
            line += f" {r_mean:6.4f}±{r_std:.4f}"
        else:
            line += "    (skipped)"
        print(line)
    print()
    print("(C2ST closer to 0.5 = better)")


def run_benchmark(
    task_names: list[str], cfg_overrides: dict | None = None
) -> list[dict]:
    """Run TabPFN-NPE on multiple tasks and produce summary."""
    all_results = []

    for i, task_name in enumerate(task_names, 1):
        print(f"\n{'#' * 72}")
        print(f"# Task {i}/{len(task_names)}: {task_name}")
        print(f"{'#' * 72}\n")

        cfg = Config(task_name=task_name)
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)

        results = run_task(cfg)
        all_results.append(results)

    print_summary(all_results)

    # Save combined results and summary plot
    out_dir = Path("pfn_testing/sbi/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_dir / "benchmark_summary.npz"),
        **{
            r["task_name"]: {k: r[k] for k in ("c2st_tabpfn", "c2st_raw") if k in r}
            for r in all_results
        },
    )

    if len(all_results) > 1:
        plot_summary_table(all_results, str(out_dir / "benchmark_summary.png"))

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Layer sweep
# ═══════════════════════════════════════════════════════════════════════════════

N_TABPFN_LAYERS = 12


def run_layer_sweep(
    task_names: list[str],
    cfg_overrides: dict | None = None,
) -> dict[str, list[dict]]:
    """Sweep all 12 TabPFN layers on each task.

    Reuses simulation data and the raw-features baseline across layers
    to avoid redundant computation.

    Returns:
        {task_name: [{"layer": int, "c2st_tabpfn": [...], "c2st_raw": [...]}]}
    """
    all_sweep_results: dict[str, list[dict]] = {}

    for ti, task_name in enumerate(task_names, 1):
        print(f"\n{'#' * 72}")
        print(f"# Layer sweep {ti}/{len(task_names)}: {task_name}")
        print(f"{'#' * 72}\n")

        cfg = Config(task_name=task_name)
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)

        # ── Simulate once ───────────────────────────────────────────────────
        print(f"[1] Simulating {cfg.n_train + cfg.n_val} samples...")
        data = simulate(cfg.task_name, cfg.n_train, cfg.n_val, cfg.seed)
        task = data["task"]
        dim_theta, dim_x = data["dim_theta"], data["dim_x"]
        print(f"  Task: {task_name} (dim_theta={dim_theta}, dim_x={dim_x})")

        defaults = get_flow_defaults(dim_theta)
        n_transforms = cfg.n_transforms or defaults["n_transforms"]
        hidden_features = cfg.hidden_features or defaults["hidden_features"]

        # ── Raw baseline (train once) ───────────────────────────────────────
        print(f"\n[2] Training raw-features baseline...")
        flow_raw = build_flow(dim_theta, dim_x, n_transforms, hidden_features, cfg.n_bins, cfg.flow_type)
        history_raw = train_flow(
            flow_raw, data["thetas_train"], data["xs_train"],
            data["thetas_val"], data["xs_val"], cfg,
        )

        def raw_sample_fn(x_obs: np.ndarray) -> np.ndarray:
            return sample_posterior(
                flow_raw, x_obs,
                history_raw["theta_mean"], history_raw["theta_std"],
                cfg.n_posterior_samples,
            )

        print("  Evaluating raw baseline...")
        c2st_raw = evaluate_posterior(
            task, raw_sample_fn, cfg.n_reference_observations,
            cfg.n_posterior_samples,
        )
        print(f"  Raw C2ST: {np.mean(c2st_raw):.4f} +/- {np.std(c2st_raw):.4f}")

        # ── Sweep layers ────────────────────────────────────────────────────
        layer_results = []

        for layer_idx in range(N_TABPFN_LAYERS):
            print(f"\n[3.{layer_idx}] Layer {layer_idx}/{N_TABPFN_LAYERS - 1}")

            cfg.layer = layer_idx
            out = cfg.output_dir
            (out / "plots").mkdir(parents=True, exist_ok=True)
            (out / "flows").mkdir(parents=True, exist_ok=True)
            (out / "results").mkdir(parents=True, exist_ok=True)

            # Extract embeddings from this layer
            embedder = TabPFNEmbedder(
                context_size=cfg.context_size, seed=cfg.seed,
                label_strategy=cfg.label_strategy,
                layer=layer_idx,
                model_type=cfg.model_type,
                device=cfg.device,
                pooling=cfg.pooling,
            )
            embedder.fit(data["xs_train"], thetas=data["thetas_train"])

            emb_train = embedder.transform(data["xs_train"])
            emb_val = embedder.transform(data["xs_val"])
            emb_dim = emb_train.shape[1]

            # Optional dimensionality reduction
            reducer = None
            if cfg.projection_checkpoint is not None:
                reducer = LinearProjectionReducer.load(cfg.projection_checkpoint)
                emb_train = reducer.transform(emb_train)
                emb_val = reducer.transform(emb_val)
                emb_dim = reducer.n_components
            elif cfg.embed_dim is not None:
                if cfg.reduction == "pca":
                    reducer = PCAReducer(cfg.embed_dim)
                elif cfg.reduction == "linear":
                    reducer = LinearProjectionReducer(cfg.embed_dim)
                elif cfg.reduction == "truncate":
                    emb_train = emb_train[:, :cfg.embed_dim]
                    emb_val = emb_val[:, :cfg.embed_dim]
                    emb_dim = cfg.embed_dim
                elif cfg.reduction in ("joint_linear", "joint_linear_pca"):
                    pass  # Projection happens inside the flow
                else:
                    raise ValueError(f"Unknown reduction: {cfg.reduction}")
                if reducer is not None:
                    emb_train = reducer.fit_transform(emb_train)
                    emb_val = reducer.transform(emb_val)
                    emb_dim = cfg.embed_dim

            # Train flow
            if cfg.reduction in ("joint_linear", "joint_linear_pca") and cfg.embed_dim is not None:
                inner_flow = build_flow(dim_theta, cfg.embed_dim, n_transforms, hidden_features, cfg.n_bins, cfg.flow_type)
                emb_mean = torch.tensor(emb_train.mean(axis=0), dtype=torch.float32)
                emb_std = torch.tensor(emb_train.std(axis=0) + 1e-8, dtype=torch.float32)
                pca_components = None
                if cfg.reduction == "joint_linear_pca":
                    standardized = (emb_train - emb_train.mean(axis=0)) / (emb_train.std(axis=0) + 1e-8)
                    pca = PCA(n_components=cfg.embed_dim)
                    pca.fit(standardized)
                    pca_components = torch.tensor(pca.components_, dtype=torch.float32)
                flow_tabpfn = ProjectedFlow(emb_dim, cfg.embed_dim, inner_flow, emb_mean, emb_std, pca_components)
            else:
                flow_tabpfn = build_flow(dim_theta, emb_dim, n_transforms, hidden_features, cfg.n_bins, cfg.flow_type)
            history_tabpfn = train_flow(
                flow_tabpfn, data["thetas_train"], emb_train,
                data["thetas_val"], emb_val, cfg,
            )

            # Evaluate
            def make_tabpfn_sample_fn(emb, red, flow, hist):
                def fn(x_obs: np.ndarray) -> np.ndarray:
                    obs_emb = emb.transform(x_obs.reshape(1, -1))
                    if red is not None:
                        obs_emb = red.transform(obs_emb)
                    elif cfg.reduction == "truncate" and cfg.embed_dim is not None:
                        obs_emb = obs_emb[:, :cfg.embed_dim]
                    return sample_posterior(
                        flow, obs_emb[0],
                        hist["theta_mean"], hist["theta_std"],
                        cfg.n_posterior_samples,
                    )
                return fn

            tabpfn_sample_fn = make_tabpfn_sample_fn(
                embedder, reducer, flow_tabpfn, history_tabpfn,
            )
            c2st_tabpfn = evaluate_posterior(
                task, tabpfn_sample_fn, cfg.n_reference_observations,
                cfg.n_posterior_samples,
            )

            result = {
                "layer": layer_idx,
                "c2st_tabpfn": c2st_tabpfn,
                "c2st_raw": c2st_raw,
                "emb_dim": emb_dim,
            }
            layer_results.append(result)

            # Save per-layer results
            np.savez(str(out / "results" / "results.npz"), **{
                k: np.array(v) if isinstance(v, list) else v
                for k, v in result.items()
            })
            if flow_tabpfn is not None:
                torch.save(flow_tabpfn.state_dict(), str(out / "flows" / "flow_tabpfn.pt"))

            print(f"  Layer {layer_idx} C2ST: {np.mean(c2st_tabpfn):.4f} +/- {np.std(c2st_tabpfn):.4f}")

        all_sweep_results[task_name] = layer_results

        # ── Print layer sweep summary ───────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"Layer sweep summary: {task_name}")
        print(f"{'=' * 60}")
        print(f"  Raw baseline C2ST: {np.mean(c2st_raw):.4f} +/- {np.std(c2st_raw):.4f}")
        print(f"  {'Layer':>5s}  {'TabPFN C2ST':>14s}")
        print(f"  {'-' * 25}")
        for r in layer_results:
            m = np.mean(r["c2st_tabpfn"])
            s = np.std(r["c2st_tabpfn"])
            best = " <-- best" if m == min(np.mean(lr["c2st_tabpfn"]) for lr in layer_results) else ""
            print(f"  {r['layer']:5d}  {m:.4f}±{s:.4f}{best}")

    # ── Save combined sweep results and plot ────────────────────────────────
    out_dir = Path("pfn_testing/sbi/outputs/layer_ablation/sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_data = {}
    for task_name, layer_results in all_sweep_results.items():
        sweep_data[task_name] = {
            "layers": [r["layer"] for r in layer_results],
            "c2st_tabpfn_means": [float(np.mean(r["c2st_tabpfn"])) for r in layer_results],
            "c2st_tabpfn_stds": [float(np.std(r["c2st_tabpfn"])) for r in layer_results],
            "c2st_raw_mean": float(np.mean(layer_results[0]["c2st_raw"])),
            "c2st_raw_std": float(np.std(layer_results[0]["c2st_raw"])),
        }

    seed = int((cfg_overrides or {}).get("seed", 42))
    pooling = (cfg_overrides or {}).get("pooling", "target")
    pool_tag = "" if pooling == "target" else f"_{pooling}"
    task_tag = "-".join(task_names) if len(task_names) <= 3 else f"{len(task_names)}tasks"
    stem = f"summary_{task_tag}{pool_tag}_s{seed}"
    np.savez(str(out_dir / f"{stem}.npz"), **sweep_data)
    plot_layer_sweep(all_sweep_results, str(out_dir / f"{stem}.png"))

    return all_sweep_results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TabPFN Embeddings as Summary Statistics for SBI",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="two_moons",
        help=f"SBIBM task name (choices: {', '.join(AVAILABLE_TASKS)})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available SBIBM tasks",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=10_000,
        help="Number of training simulations (default: 10000)",
    )
    parser.add_argument(
        "--nval",
        type=int,
        default=2_000,
        help="Number of validation simulations (default: 2000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device override for TabPFN embeddings (default: auto)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Target embedding dimension after reduction (default: no reduction)",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="pca",
        choices=["pca", "linear", "truncate", "joint_linear", "joint_linear_pca"],
        help="Embedding reduction method (default: pca)",
    )
    parser.add_argument(
        "--label-strategy",
        type=str,
        default="random",
        choices=["random", "constant", "theta_pca", "per_dim", "per_dim_mean"],
        help="Label strategy for TabPFN context (default: random)",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="TabPFN transformer layer to extract embeddings from (0-11, default: last)",
    )
    parser.add_argument(
        "--layer-sweep", action="store_true",
        help="Sweep all 12 TabPFN layers on the specified task(s)",
    )
    parser.add_argument(
        "--pooling", type=str, default="target",
        choices=["target", "mean", "first"],
        help="Feature-axis pooling for hook-based extraction (default: target = [:,:,-1])",
    )
    parser.add_argument(
        "--model-type", type=str, default="classifier",
        choices=["classifier", "regressor"],
        help="TabPFN model type (default: classifier)",
    )
    parser.add_argument(
        "--projection-checkpoint", type=str, default=None,
        help="Path to a pretrained LinearProjectionReducer checkpoint (.pt)",
    )
    parser.add_argument(
        "--projection-sweep", action="store_true",
        help="Sweep all pretrained projections in pretrained/{task}/ directory",
    )
    parser.add_argument(
        "--projection-dir", type=str, default="pfn_testing/sbi/pretrained",
        help="Directory containing pretrained projections (default: pfn_testing/sbi/pretrained)",
    )
    parser.add_argument(
        "--finetune-layers", type=int, default=0,
        help="Fine-tune last N TabPFN transformer layers (0=frozen, default: 0)",
    )
    parser.add_argument(
        "--encoder-checkpoint", type=str, default=None,
        help="Path to contrastive-pretrained encoder checkpoint (.pt)",
    )
    parser.add_argument(
        "--mrl-dim", type=int, default=None,
        help="MRL: truncate each per-dim embedding to N dims before concatenation",
    )
    parser.add_argument(
        "--sbc", action="store_true",
        help="Run Simulation-Based Calibration after C2ST evaluation",
    )
    parser.add_argument(
        "--sbc-trials", type=int, default=1000,
        help="Number of SBC trials (default: 1000)",
    )
    parser.add_argument(
        "--gloeckler-decomp",
        action="store_true",
        help="AR1 only: use compositional transition-based TabPFN embeddings",
    )
    parser.add_argument(
        "--flow-type", type=str, default="nsf",
        choices=["nsf", "naf", "fmpe"],
        help="Density estimator type: nsf (default), naf, or fmpe",
    )
    parser.add_argument(
        "--n-transforms", type=int, default=None,
        help="Number of flow transforms (default: auto based on dim_theta)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=8,
        help="Number of spline bins per transform (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to pre-generated .npz dataset (skips simulation)",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="Wandb project name (disabled if not set)",
    )
    parser.add_argument(
        "--skip-raw", action="store_true",
        help="Skip raw-features baseline (only train/eval TabPFN flow)",
    )
    args = parser.parse_args()
    print(f"Using device: {args.device}")

    overrides = {"n_train": args.n_train, "n_val": args.nval, "seed": args.seed}
    if args.device != "auto":
        overrides["device"] = args.device
    if args.embed_dim is not None:
        overrides["embed_dim"] = args.embed_dim
        overrides["reduction"] = args.reduction
    if args.projection_checkpoint is not None:
        overrides["projection_checkpoint"] = args.projection_checkpoint
    if args.label_strategy != "random":
        overrides["label_strategy"] = args.label_strategy
    if args.layer is not None:
        overrides["layer"] = args.layer
    if args.model_type != "classifier":
        overrides["model_type"] = args.model_type
    if args.pooling != "target":
        overrides["pooling"] = args.pooling
    if args.finetune_layers > 0:
        overrides["finetune_layers"] = args.finetune_layers
    if args.encoder_checkpoint is not None:
        overrides["encoder_checkpoint"] = args.encoder_checkpoint
    if args.mrl_dim is not None:
        overrides["mrl_dim"] = args.mrl_dim
    if args.sbc:
        overrides["run_sbc"] = True
        overrides["sbc_trials"] = args.sbc_trials
    if args.gloeckler_decomp:
        overrides["gloeckler_decomp"] = True
    if args.flow_type != "nsf":
        overrides["flow_type"] = args.flow_type
    if args.n_transforms is not None:
        overrides["n_transforms"] = args.n_transforms
    if args.n_bins != 8:
        overrides["n_bins"] = args.n_bins
    if args.data_path is not None:
        overrides["data_path"] = args.data_path
    if args.wandb_project is not None:
        overrides["wandb_project"] = args.wandb_project
    if args.skip_raw:
        overrides["skip_raw"] = True

    task_names = AVAILABLE_TASKS if args.all else [args.task]

    if args.projection_sweep:
        # Discover and sweep all pretrained checkpoints per task
        proj_dir = Path(args.projection_dir)
        for task_name in task_names:
            task_proj_dir = proj_dir / task_name
            checkpoints = sorted(task_proj_dir.glob("linear_*.pt"))
            if not checkpoints:
                print(f"No pretrained checkpoints found in {task_proj_dir}, skipping {task_name}")
                continue
            print(f"\nProjection sweep for {task_name}: {[c.name for c in checkpoints]}")
            for ckpt_path in checkpoints:
                sweep_overrides = {**overrides, "projection_checkpoint": str(ckpt_path)}
                run_benchmark([task_name], sweep_overrides)
    elif args.layer_sweep:
        run_layer_sweep(task_names, overrides)
    else:
        run_benchmark(task_names, overrides)


if __name__ == "__main__":
    main()
