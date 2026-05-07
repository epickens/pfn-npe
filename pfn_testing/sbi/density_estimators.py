"""Density estimators for conditional posterior estimation.

Supports:
  - NSF (Neural Spline Flow) — default, via zuko
  - NAF (Neural Autoregressive Flow) — universal density approximator, via zuko
  - FMPE (Flow Matching Posterior Estimation) — via sbi library
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import zuko.flows

if TYPE_CHECKING:
    from pfn_testing.sbi.tabpfn_npe import Config


# ═══════════════════════════════════════════════════════════════════════════════
# Flow architecture defaults
# ═══════════════════════════════════════════════════════════════════════════════


def get_flow_defaults(dim_theta: int) -> dict:
    """Dimension-aware flow architecture defaults."""
    if dim_theta <= 5:
        return {"n_transforms": 5, "hidden_features": [128, 128]}
    else:
        return {"n_transforms": 8, "hidden_features": [256, 256]}


# ═══════════════════════════════════════════════════════════════════════════════
# Flow wrappers
# ═══════════════════════════════════════════════════════════════════════════════


class ProjectedFlow(nn.Module):
    """Flow with a jointly-trained linear projection on the context."""

    def __init__(self, d_embedding, d_projected, flow, emb_mean, emb_std, pca_components=None):
        super().__init__()
        self.projector = nn.Linear(d_embedding, d_projected)
        self.flow = flow
        self.register_buffer("emb_mean", emb_mean)
        self.register_buffer("emb_std", emb_std)
        if pca_components is not None:
            with torch.no_grad():
                self.projector.weight.copy_(pca_components)
                self.projector.bias.zero_()

    def forward(self, context):
        z = (context - self.emb_mean) / self.emb_std
        return self.flow(self.projector(z))


class FineTunedFlow(nn.Module):
    """End-to-end differentiable: TabPFN embedder → linear projector → flow."""

    def __init__(self, embedder, d_embedding, d_projected, flow, emb_mean, emb_std):
        super().__init__()
        self.embedder = embedder
        self.projector = nn.Linear(d_embedding, d_projected)
        self.flow = flow
        self.register_buffer("emb_mean", emb_mean)
        self.register_buffer("emb_std", emb_std)

    def forward(self, context):
        emb = self.embedder(context)
        z = (emb - self.emb_mean) / self.emb_std
        return self.flow(self.projector(z))


# ═══════════════════════════════════════════════════════════════════════════════
# Flow construction
# ═══════════════════════════════════════════════════════════════════════════════


def build_flow(
    dim_theta: int,
    dim_context: int,
    n_transforms: int = 5,
    hidden_features: list[int] | None = None,
    n_bins: int = 8,
    flow_type: str = "nsf",
) -> nn.Module:
    """Build a conditional normalizing flow.

    Args:
        flow_type: "nsf" (Neural Spline Flow) or "naf" (Neural Autoregressive Flow).
    """
    if hidden_features is None:
        hidden_features = [128, 128]

    if flow_type == "nsf":
        return zuko.flows.NSF(
            features=dim_theta,
            context=dim_context,
            transforms=n_transforms,
            bins=n_bins,
            hidden_features=hidden_features,
        )
    elif flow_type == "naf":
        return zuko.flows.NAF(
            features=dim_theta,
            context=dim_context,
            transforms=n_transforms,
            signal=max(dim_context, 16),
            hidden_features=hidden_features,
        )
    else:
        raise ValueError(f"Unknown flow_type for build_flow: {flow_type}. Use 'nsf' or 'naf'.")


# ═══════════════════════════════════════════════════════════════════════════════
# Training (zuko flows)
# ═══════════════════════════════════════════════════════════════════════════════


def train_flow(
    flow: nn.Module,
    thetas_train: np.ndarray,
    context_train: np.ndarray,
    thetas_val: np.ndarray,
    context_val: np.ndarray,
    cfg: Config,
) -> dict:
    """Train a conditional normalizing flow (NSF or NAF).

    Returns dict with training history and standardization parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow = flow.to(device)

    # Standardize theta
    theta_mean = thetas_train.mean(axis=0)
    theta_std = thetas_train.std(axis=0) + 1e-8
    thetas_train_n = (thetas_train - theta_mean) / theta_std
    thetas_val_n = (thetas_val - theta_mean) / theta_std

    theta_train_t = torch.tensor(thetas_train_n, dtype=torch.float32, device=device)
    ctx_train_t = torch.tensor(context_train, dtype=torch.float32, device=device)
    theta_val_t = torch.tensor(thetas_val_n, dtype=torch.float32, device=device)
    ctx_val_t = torch.tensor(context_val, dtype=torch.float32, device=device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(theta_train_t, ctx_train_t),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(flow.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs
    )

    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve = 0
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "theta_mean": theta_mean,
        "theta_std": theta_std,
    }

    for epoch in range(cfg.max_epochs):
        flow.train()
        epoch_losses = []
        for theta_b, ctx_b in loader:
            theta_b, ctx_b = theta_b.to(device), ctx_b.to(device)
            optimizer.zero_grad()
            log_prob = flow(ctx_b).log_prob(theta_b)
            loss = -log_prob.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()

        train_loss = float(np.mean(epoch_losses))

        flow.eval()
        with torch.no_grad():
            val_loss = -flow(ctx_val_t).log_prob(theta_val_t).mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if no_improve >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    assert best_state is not None
    flow.load_state_dict(best_state)
    history["best_epoch"] = epoch - no_improve
    return history


def train_finetuned_flow(
    flow_model: FineTunedFlow,
    thetas_train: np.ndarray,
    xs_train_pp: np.ndarray,
    thetas_val: np.ndarray,
    xs_val_pp: np.ndarray,
    cfg: Config,
) -> dict:
    """Train FineTunedFlow end-to-end (TabPFN layers + projector + flow).

    Uses separate LR for TabPFN params vs projector/flow params.
    """
    try:
        import wandb
    except ImportError:
        wandb = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_model = flow_model.to(device)

    # Standardize theta
    theta_mean = thetas_train.mean(axis=0)
    theta_std = thetas_train.std(axis=0) + 1e-8
    thetas_train_n = (thetas_train - theta_mean) / theta_std
    thetas_val_n = (thetas_val - theta_mean) / theta_std

    theta_train_t = torch.tensor(thetas_train_n, dtype=torch.float32, device=device)
    xs_train_t = torch.tensor(xs_train_pp, dtype=torch.float32, device=device)
    theta_val_t = torch.tensor(thetas_val_n, dtype=torch.float32, device=device)
    xs_val_t = torch.tensor(xs_val_pp, dtype=torch.float32, device=device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(theta_train_t, xs_train_t),
        batch_size=cfg.ft_batch_size,
        shuffle=True,
    )

    # Separate param groups: TabPFN unfrozen params get lower LR
    tabpfn_params = [p for p in flow_model.embedder.model.parameters() if p.requires_grad]
    other_params = list(flow_model.projector.parameters()) + list(flow_model.flow.parameters())
    optimizer = torch.optim.Adam([
        {"params": tabpfn_params, "lr": cfg.lr_tabpfn},
        {"params": other_params, "lr": cfg.lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve = 0
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "theta_mean": theta_mean,
        "theta_std": theta_std,
    }

    for epoch in range(cfg.max_epochs):
        flow_model.train()
        epoch_losses = []
        for theta_b, xs_b in loader:
            optimizer.zero_grad()
            log_prob = flow_model(xs_b).log_prob(theta_b)
            loss = -log_prob.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()

        train_loss = float(np.mean(epoch_losses))

        flow_model.eval()
        with torch.no_grad():
            val_loss = -flow_model(xs_val_t).log_prob(theta_val_t).mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in flow_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if wandb is not None and wandb.run:
            wandb.log({"flow/train_loss": train_loss, "flow/val_loss": val_loss, "epoch": epoch})

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if no_improve >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    assert best_state is not None
    flow_model.load_state_dict(best_state)
    history["best_epoch"] = epoch - no_improve
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling (zuko flows)
# ═══════════════════════════════════════════════════════════════════════════════


def sample_posterior(
    flow: nn.Module,
    context: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    n_samples: int = 10_000,
) -> np.ndarray:
    """Sample from flow posterior and un-standardize."""
    flow.eval()
    device = next(flow.parameters()).device
    ctx_t = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        samples = flow(ctx_t).sample((n_samples,))
        samples = samples.squeeze(1).cpu().numpy()

    return samples * theta_std + theta_mean


# ═══════════════════════════════════════════════════════════════════════════════
# FMPE (Flow Matching Posterior Estimation)
# ═══════════════════════════════════════════════════════════════════════════════


def train_fmpe(
    thetas_train: np.ndarray,
    context_train: np.ndarray,
    thetas_val: np.ndarray,
    context_val: np.ndarray,
    cfg: Config,
) -> tuple:
    """Train FMPE and return (posterior, history_dict).

    Uses sbi's FMPE with a wide BoxUniform prior on standardized theta.
    """
    from sbi.inference import FMPE
    from sbi.utils import BoxUniform

    dim_theta = thetas_train.shape[1]
    theta_mean = thetas_train.mean(axis=0)
    theta_std = thetas_train.std(axis=0) + 1e-8
    thetas_n = (thetas_train - theta_mean) / theta_std

    # sbi needs a torch prior — wide uniform over standardized space
    sbi_prior = BoxUniform(
        low=torch.full((dim_theta,), -10.0),
        high=torch.full((dim_theta,), 10.0),
    )

    fmpe = FMPE(prior=sbi_prior)
    fmpe.append_simulations(
        torch.tensor(thetas_n, dtype=torch.float32),
        torch.tensor(context_train, dtype=torch.float32),
    )

    print(f"  Training FMPE (batch_size={cfg.batch_size}, max_epochs={cfg.max_epochs})...")
    fmpe.train(
        training_batch_size=cfg.batch_size,
        max_num_epochs=cfg.max_epochs,
    )
    posterior = fmpe.build_posterior()

    return posterior, {"theta_mean": theta_mean, "theta_std": theta_std}


def sample_fmpe_posterior(
    posterior,
    context: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    n_samples: int = 10_000,
) -> np.ndarray:
    """Sample from FMPE posterior and un-standardize."""
    ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    samples = posterior.sample((n_samples,), x=ctx).numpy()
    return samples * theta_std + theta_mean
