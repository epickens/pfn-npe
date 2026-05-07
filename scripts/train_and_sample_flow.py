"""Train and sample a non-NSF density estimator on cached TabPFN embeddings.

Supports `--flow-type {fmpe, mixture_nsf}` for direct comparison against the
existing default-NSF flows. Trains the chosen estimator, samples 1000
posterior draws for each of the 10 sbibm reference observations, and saves
to `flow_vs_quantile/{task}_s{seed}_{flow_type}.npz` with the same fields
the existing diagnostics (`compare_flow_vs_quantile.py`,
`c2st_decomposition.py`, `count_modes.py`) consume.

For `mixture_nsf`, defines a small `MixtureFlow` (K conditional NSFs +
context-conditioned gating MLP); training reuses the existing `train_flow`
path because the wrapper exposes the same `flow(context).log_prob(theta)`
and `flow(context).sample(...)` interface as zuko flows.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow, get_flow_defaults, sample_fmpe_posterior, sample_posterior,
    train_flow, train_fmpe,
)
from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import PCAReducer, TabPFNEmbedder  # noqa: E402
from scripts.layer_quantile_probe import _pinball_np  # noqa: E402

OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")


@dataclass
class TrainConfig:
    """Tiny config — only the fields `train_flow` / `train_fmpe` actually read."""
    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 5.0
    flow_type: str = "nsf"
    hidden_features: list[int] = field(default_factory=lambda: [128, 128])


class MixtureDistribution:
    """Conditional log_prob / sample wrapper that mimics zuko's flow(ctx)."""

    def __init__(self, mixture: "MixtureFlow", ctx: torch.Tensor) -> None:
        self.mixture = mixture
        self.ctx = ctx               # (n_batch, dim_context)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        gate_logits = self.mixture.gate(self.ctx)                # (n_batch, K)
        log_pi = F.log_softmax(gate_logits, dim=-1)
        log_p = torch.stack(
            [self.mixture.flows[k](self.ctx).log_prob(theta)
             for k in range(self.mixture.K)],
            dim=-1,
        )                                                          # (n_batch, K)
        return torch.logsumexp(log_pi + log_p, dim=-1)

    def sample(self, sample_shape: tuple[int, ...]) -> torch.Tensor:
        n_samples = sample_shape[0]
        ctx = self.ctx                                             # expect (1, D)
        if ctx.dim() == 2 and ctx.shape[0] != 1:
            raise ValueError("MixtureDistribution.sample expects single-context (1, D)")
        gate_logits = self.mixture.gate(ctx)
        pi = F.softmax(gate_logits, dim=-1).squeeze(0)             # (K,)
        ks = torch.multinomial(pi, n_samples, replacement=True)
        out = torch.empty(n_samples, ctx.shape[0], self.mixture.dim_theta,
                          device=ctx.device)
        for k in range(self.mixture.K):
            mask = ks == k
            n_k = int(mask.sum().item())
            if n_k == 0:
                continue
            samples_k = self.mixture.flows[k](ctx).sample((n_k,))
            out[mask] = samples_k
        return out


class MixtureFlow(nn.Module):
    """Conditional mixture of K NSFs with context-conditioned gating."""

    def __init__(
        self,
        dim_theta: int,
        dim_context: int,
        K: int = 2,
        n_transforms: int = 5,
        hidden_features: list[int] | None = None,
        n_bins: int = 8,
    ) -> None:
        super().__init__()
        self.K = K
        self.dim_theta = dim_theta
        if hidden_features is None:
            hidden_features = [128, 128]
        self.flows = nn.ModuleList([
            zuko.flows.NSF(
                features=dim_theta, context=dim_context,
                transforms=n_transforms, bins=n_bins,
                hidden_features=hidden_features,
            )
            for _ in range(K)
        ])
        gate_layers: list[nn.Module] = [nn.Linear(dim_context, hidden_features[0]),
                                        nn.ReLU()]
        for h_in, h_out in zip(hidden_features[:-1], hidden_features[1:], strict=True):
            gate_layers.extend([nn.Linear(h_in, h_out), nn.ReLU()])
        gate_layers.append(nn.Linear(hidden_features[-1], K))
        self.gate = nn.Sequential(*gate_layers)

    def forward(self, ctx: torch.Tensor) -> MixtureDistribution:
        return MixtureDistribution(self, ctx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flow-type",
                    choices=["fmpe", "mixture_nsf", "nsf", "nsf_large", "nsf_xl"],
                    required=True)
    ap.add_argument("--K", type=int, default=2,
                    help="Number of mixture components (mixture_nsf only)")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-flow-samples", type=int, default=1000)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--context-size", type=int, default=1000,
                    help="TabPFN context size used for PFN-NPE embeddings.")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"],
                    help="TabPFN model version for the encoder. Adds an "
                         "_mv25 suffix to the output filename when non-default "
                         "so v2.5 runs don't clobber v2 cells.")
    ap.add_argument("--embed-dim", type=int, default=None,
                    help="Optional embedding dimension after reduction, e.g. 64 "
                         "to match TabPFN-NPE PCA-64 runs.")
    ap.add_argument("--reduction", choices=["pca", "truncate"], default="pca",
                    help="Embedding reduction used when --embed-dim is set.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="Device override for TabPFN embedding extraction.")
    ap.add_argument("--encoder-checkpoint", default=None,
                    help="Path to contrastive-finetuned encoder checkpoint .pt; "
                         "loaded after embedder.fit().")
    ap.add_argument("--encoder-tag", default=None,
                    help="Short label for the encoder, used in output filename "
                         "(e.g. 'cl_step190k'). Defaults to checkpoint stem.")
    ap.add_argument("--normalize-emb", action="store_true",
                    help="Per-dim z-score embeddings (using train statistics) "
                         "before passing to NSF / FMPE / mixture. Useful if the "
                         "encoder produces a different per-dim scale than raw "
                         "TabPFN.")
    ap.add_argument("--output-method", default=None,
                    help="Override the method tag used in the output filename "
                         "and saved flow_type metadata. This is useful for "
                         "ablation reruns that share the same estimator but "
                         "must not collide with canonical outputs.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"mixture_nsf_K{args.K}" if args.flow_type == "mixture_nsf"
        else args.flow_type
    )
    if args.output_method:
        suffix = args.output_method
    if args.n_train != 10000:
        suffix = f"{suffix}_n{args.n_train}"
    if args.context_size != 1000:
        suffix = f"{suffix}_ctx{args.context_size}"
    if args.embed_dim is not None:
        if args.reduction == "pca":
            suffix = f"{suffix}_pca{args.embed_dim}"
        else:
            suffix = f"{suffix}_truncate{args.embed_dim}"
    if args.encoder_checkpoint:
        enc_label = args.encoder_tag or Path(args.encoder_checkpoint).stem
        suffix = f"{suffix}_enc_{enc_label}"
    if args.normalize_emb:
        suffix = f"{suffix}_zscore"
    if args.model_version != "v2":
        # Match the layer_linear_probe / cross_theta_probe convention: `_mv25`.
        suffix = f"{suffix}_m{args.model_version.replace('.', '')}"
    out_npz = OUT_DIR / f"{args.task}_s{args.seed}_{suffix}.npz"
    if out_npz.exists():
        print(f"[skip] {out_npz}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"task={args.task} seed={args.seed} flow_type={suffix} (deterministic)")
    t_train_start = time.perf_counter()
    print("Simulating training data...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = data["dim_theta"]
    dim_x = data["dim_x"]
    print(f"  dim_theta={dim_theta}, dim_x={dim_x}")

    print(f"Embedding train/val/ref at layer=None (model={args.model_version})...")
    emb = TabPFNEmbedder(
        context_size=args.context_size, seed=args.seed, label_strategy="per_dim",
        layer=None, model_type="regressor", device=args.device,
        model_version=args.model_version,
    )
    emb.fit(data["xs_train"], thetas=data["thetas_train"])
    if args.encoder_checkpoint:
        print(f"Loading contrastive encoder: {args.encoder_checkpoint}")
        emb.load_encoder_checkpoint(args.encoder_checkpoint)
    emb_train = emb.transform(data["xs_train"])
    emb_val = emb.transform(data["xs_val"])

    task = get_task(args.task)
    x_ref = np.stack([
        task.get_observation(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    theta_true = np.stack([
        task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
        for i in range(args.n_ref)
    ])
    e_ref = emb.transform(x_ref)
    print(f"  emb_train {emb_train.shape}, e_ref {e_ref.shape}")

    if args.embed_dim is not None:
        if args.reduction == "pca":
            reducer = PCAReducer(args.embed_dim)
            print(f"  Reducing embeddings with PCA -> {args.embed_dim}d")
            emb_train = reducer.fit_transform(emb_train)
            emb_val = reducer.transform(emb_val)
            e_ref = reducer.transform(e_ref)
        elif args.reduction == "truncate":
            print(f"  Truncating embeddings -> first {args.embed_dim} dims")
            emb_train = emb_train[:, :args.embed_dim]
            emb_val = emb_val[:, :args.embed_dim]
            e_ref = e_ref[:, :args.embed_dim]
        else:
            raise ValueError(f"Unknown reduction: {args.reduction!r}")
        print(f"  reduced emb_train {emb_train.shape}, e_ref {e_ref.shape}")

    if args.normalize_emb:
        emb_train = emb_train.astype(np.float32, copy=False)
        emb_val = emb_val.astype(np.float32, copy=False)
        e_ref = e_ref.astype(np.float32, copy=False)
        emb_mean = emb_train.mean(axis=0)
        emb_std = emb_train.std(axis=0) + 1e-6
        emb_train = (emb_train - emb_mean) / emb_std
        emb_val = (emb_val - emb_mean) / emb_std
        e_ref = (e_ref - emb_mean) / emb_std
        print(f"  z-scored embedding (train mean abs={float(np.abs(emb_mean).mean()):.3f}, "
              f"train std mean={float(emb_std.mean()):.3f})")

    cfg = TrainConfig(
        lr=args.lr, max_epochs=args.max_epochs,
        flow_type=args.flow_type,
    )

    if args.flow_type == "fmpe":
        print("\nTraining FMPE...")
        posterior, history = train_fmpe(
            data["thetas_train"], emb_train,
            data["thetas_val"], emb_val, cfg,
        )

        def sample_one(ctx_row: np.ndarray) -> np.ndarray:
            return sample_fmpe_posterior(
                posterior, ctx_row,
                history["theta_mean"], history["theta_std"],
                args.n_flow_samples,
            )
    elif args.flow_type in ("nsf", "nsf_large", "nsf_xl"):
        defaults = get_flow_defaults(dim_theta)
        if args.flow_type == "nsf":
            n_t, hidden, n_bins = defaults["n_transforms"], defaults["hidden_features"], 8
        elif args.flow_type == "nsf_large":
            n_t, hidden, n_bins = 12, [512, 512], 16
        else:
            n_t, hidden, n_bins = 20, [512, 512], 16
        print(f"\nBuilding NSF (n_transforms={n_t}, hidden={hidden}, n_bins={n_bins})...")
        flow = build_flow(
            dim_theta=dim_theta, dim_context=emb_train.shape[1],
            n_transforms=n_t, hidden_features=hidden, n_bins=n_bins,
        )
        n_params = sum(p.numel() for p in flow.parameters())
        print(f"  {n_params} params total")
        history = train_flow(
            flow,
            data["thetas_train"], emb_train,
            data["thetas_val"], emb_val, cfg,
        )

        def sample_one(ctx_row: np.ndarray) -> np.ndarray:
            return sample_posterior(
                flow, ctx_row,
                history["theta_mean"], history["theta_std"],
                args.n_flow_samples,
            )
    else:
        defaults = get_flow_defaults(dim_theta)
        print(f"\nBuilding MixtureFlow (K={args.K}, "
              f"n_transforms={defaults['n_transforms']}, "
              f"hidden={defaults['hidden_features']}, n_bins=8)...")
        mixture = MixtureFlow(
            dim_theta=dim_theta, dim_context=emb_train.shape[1],
            K=args.K,
            n_transforms=defaults["n_transforms"],
            hidden_features=defaults["hidden_features"],
            n_bins=8,
        )
        n_params = sum(p.numel() for p in mixture.parameters())
        print(f"  {n_params} params total")
        history = train_flow(
            mixture,
            data["thetas_train"], emb_train,
            data["thetas_val"], emb_val, cfg,
        )

        def sample_one(ctx_row: np.ndarray) -> np.ndarray:
            return sample_posterior(
                mixture, ctx_row,
                history["theta_mean"], history["theta_std"],
                args.n_flow_samples,
            )

    t_train_end = time.perf_counter()
    print(f"[TIMING] phase=train duration={t_train_end - t_train_start:.3f}")

    t_sample_start = time.perf_counter()
    print("\nSampling for ref obs...")
    flow_samples = np.zeros(
        (args.n_ref, args.n_flow_samples, dim_theta), dtype=np.float32,
    )
    for i in range(args.n_ref):
        flow_samples[i] = sample_one(e_ref[i])
        print(f"  obs {i+1}: drew {flow_samples.shape[1]} samples")
    t_sample_end = time.perf_counter()
    print(f"[TIMING] phase=sample duration={t_sample_end - t_sample_start:.3f}")

    # MCMC empirical quantiles + true-θ pinball, in the same format as the
    # existing flow_vs_quantile npz so c2st_decomposition + count_modes can
    # consume the file unchanged.
    taus = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float32)
    flow_q = np.zeros((args.n_ref, len(taus), dim_theta), dtype=np.float32)
    emp_q = np.zeros_like(flow_q)
    for i in range(args.n_ref):
        flow_q[i] = np.quantile(flow_samples[i], taus, axis=0)
        ref = task.get_reference_posterior_samples(num_observation=i + 1).numpy()
        emp_q[i] = np.quantile(ref, taus, axis=0)
    flow_pinball = float(_pinball_np(
        theta_true, flow_q.transpose(1, 0, 2), taus,
    ).mean())

    save_kwargs = dict(
        taus=taus,
        flow_q=flow_q, emp_q=emp_q,
        flow_samples=flow_samples,
        theta_true=theta_true, x_ref=x_ref,
        flow_pinball=flow_pinball,
        task=args.task, seed=args.seed, flow_type=suffix,
        embed_dim=args.embed_dim if args.embed_dim is not None else emb_train.shape[1],
        reduction=args.reduction if args.embed_dim is not None else "none",
        n_train=args.n_train,
    )

    # Capture gate behavior for mixture flows.
    if args.flow_type == "mixture_nsf":
        device = next(mixture.parameters()).device
        with torch.no_grad():
            n_train_sub = min(2000, emb_train.shape[0])
            train_idx = np.random.choice(emb_train.shape[0], size=n_train_sub, replace=False)
            ctx_train_sub = torch.tensor(emb_train[train_idx], dtype=torch.float32, device=device)
            ctx_ref = torch.tensor(e_ref, dtype=torch.float32, device=device)
            pi_train = F.softmax(mixture.gate(ctx_train_sub), dim=-1).cpu().numpy()
            pi_ref = F.softmax(mixture.gate(ctx_ref), dim=-1).cpu().numpy()
        print("\nGate diagnostic:")
        print(f"  π_train shape={pi_train.shape}, "
              f"max π per row mean={pi_train.max(axis=-1).mean():.3f} "
              f"(entropy mean={(-pi_train * np.log(pi_train + 1e-12)).sum(axis=-1).mean():.3f}, "
              f"max log K={float(np.log(args.K)):.3f})")
        print(f"  π_ref shape={pi_ref.shape}, "
              f"max π per row mean={pi_ref.max(axis=-1).mean():.3f}")
        per_component_max = pi_train.max(axis=0)
        print(f"  per-component max π over train: {per_component_max}")
        save_kwargs["gate_pi_train"] = pi_train.astype(np.float32)
        save_kwargs["gate_pi_ref"] = pi_ref.astype(np.float32)
        save_kwargs["gate_K"] = args.K

    np.savez(str(out_npz), **save_kwargs)
    print(f"\nflow pinball: {flow_pinball:.4f}")
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
