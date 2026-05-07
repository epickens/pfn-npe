"""Classifier density-ratio reweighting in rank/PIT space.

This is an oracle post-hoc repair for saved posterior samples. It reads an
existing `flow_vs_quantile/{task}_s{seed}[_{method}].npz`, estimates a density
ratio between reference posterior samples and the saved samples in rank/PIT
space, and writes a new `flow_vs_quantile` npz whose `flow_samples` are
importance-resampled from the original samples.

Default behavior is copula-focused:

  * approximate samples are transformed by their own empirical marginal CDFs;
  * reference samples are transformed by their own empirical marginal CDFs;
  * a classifier is trained on Gaussianized PIT coordinates;
  * approximate samples are resampled with weights exp(classifier_logit).

With balanced classifier classes, the fitted logit estimates
log p_ref(u) - log p_base(u), so exp(logit) is the density-ratio weight.

Usage:
  uv run python scripts/rank_density_ratio_reweight.py --task slcp --seed 42
  uv run python scripts/rank_density_ratio_reweight.py \
      --task two_moons --seed 42 --input-method raw_tabq_ps1000
  uv run python scripts/run_c2st_sweep.py --include-method rank_dr --force
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task  # noqa: E402
from scripts.layer_quantile_probe import _pinball_np  # noqa: E402

FVQ_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
DEFAULT_TAUS = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float32)


class RatioMLP(nn.Module):
    """Small binary classifier whose logit is used as a density-ratio estimate."""

    def __init__(self, dim: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def source_npz_path(
    task: str,
    seed: int,
    input_method: str,
    in_dir: Path,
) -> tuple[Path, str]:
    """Resolve a source flow_vs_quantile file and normalized method tag."""
    method = input_method.strip("_")
    if method in {"", "nsf"}:
        candidates = [
            in_dir / f"{task}_s{seed}.npz",
            in_dir / f"{task}_s{seed}_nsf.npz",
        ]
        normalized = "nsf"
    else:
        candidates = [in_dir / f"{task}_s{seed}_{method}.npz"]
        normalized = method

    for path in candidates:
        if path.exists():
            return path, normalized
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No source npz found. Tried:\n  {tried}")


def empirical_pit(
    values: np.ndarray,
    reference: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Map values through the empirical CDF of reference, per dimension."""
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    denom = reference.shape[0] + 1.0
    for d in range(values.shape[1]):
        ref_sorted = np.sort(reference[:, d], kind="mergesort")
        # Average left/right ranks gives stable mid-ranks for ties.
        left = np.searchsorted(ref_sorted, values[:, d], side="left")
        right = np.searchsorted(ref_sorted, values[:, d], side="right")
        out[:, d] = (0.5 * (left + right) + 1.0) / denom
    return np.clip(out, eps, 1.0 - eps)


def pooled_pit(
    base: np.ndarray,
    ref: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map two sample sets through pooled empirical ranks per dimension."""
    n_base = base.shape[0]
    pooled = np.concatenate([base, ref], axis=0).astype(np.float64, copy=False)
    out = np.empty_like(pooled, dtype=np.float64)
    denom = pooled.shape[0] + 1.0
    for d in range(pooled.shape[1]):
        order = np.argsort(pooled[:, d], kind="mergesort")
        ranks = np.empty(pooled.shape[0], dtype=np.float64)
        ranks[order] = (np.arange(pooled.shape[0], dtype=np.float64) + 1.0) / denom
        out[:, d] = ranks
    return np.clip(out[:n_base], eps, 1.0 - eps), np.clip(out[n_base:], eps, 1.0 - eps)


def pit_features(
    base: np.ndarray,
    ref: np.ndarray,
    pit_mode: str,
    space: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return classifier features for base/ref samples in PIT or z-PIT space."""
    if pit_mode == "separate":
        base_u = empirical_pit(base, base, eps)
        ref_u = empirical_pit(ref, ref, eps)
    elif pit_mode == "base":
        base_u = empirical_pit(base, base, eps)
        ref_u = empirical_pit(ref, base, eps)
    elif pit_mode == "pooled":
        base_u, ref_u = pooled_pit(base, ref, eps)
    else:
        raise ValueError(f"Unknown pit_mode: {pit_mode}")

    if space == "u":
        return base_u.astype(np.float32), ref_u.astype(np.float32)
    if space == "z":
        return ndtri(base_u).astype(np.float32), ndtri(ref_u).astype(np.float32)
    raise ValueError(f"Unknown space: {space}")


def split_indices(n: int, val_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    n_val = int(round(n * val_frac))
    if n_val <= 0:
        return idx, idx[:0]
    if n_val >= n:
        n_val = max(1, n // 5)
    return idx[n_val:], idx[:n_val]


def train_ratio_classifier(
    base_x: np.ndarray,
    ref_x: np.ndarray,
    hidden: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    val_frac: float,
    seed: int,
) -> tuple[RatioMLP, dict[str, float]]:
    """Train a balanced base-vs-reference classifier."""
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = min(len(base_x), len(ref_x))
    base_idx = rng.choice(len(base_x), size=n, replace=False)
    ref_idx = rng.choice(len(ref_x), size=n, replace=False)

    x = np.concatenate([base_x[base_idx], ref_x[ref_idx]], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)], axis=0).astype(np.float32)

    train_idx, val_idx = split_indices(len(x), val_frac, rng)
    x_train = torch.tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=min(batch_size, len(train_idx)),
        shuffle=True,
    )

    model = RatioMLP(dim=x.shape[1], hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_val = torch.tensor(x[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    no_improve = 0

    with torch.enable_grad():
        for epoch in range(max_epochs):
            model.train()
            losses = []
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            model.eval()
            with torch.no_grad():
                if len(val_idx) > 0:
                    logits_val = model(x_val)
                    val_loss = float(F.binary_cross_entropy_with_logits(logits_val, y_val).cpu())
                else:
                    val_loss = float(np.mean(losses))

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if len(val_idx) > 0:
            logits_val = model(x_val)
            val_acc = float(((logits_val > 0) == (y_val > 0.5)).float().mean().cpu())
            val_bce = float(F.binary_cross_entropy_with_logits(logits_val, y_val).cpu())
        else:
            val_acc = math.nan
            val_bce = best_val

    return model, {
        "n_classifier_per_class": float(n),
        "n_classifier_train": float(len(train_idx)),
        "n_classifier_val": float(len(val_idx)),
        "val_bce": val_bce,
        "val_acc": val_acc,
        "best_val_bce": best_val,
    }


def classifier_log_weights(
    model: RatioMLP,
    features: np.ndarray,
    temperature: float,
    max_log_weight: float,
    batch_size: int,
) -> np.ndarray:
    """Evaluate normalized log weights on base samples."""
    device = next(model.parameters()).device
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.tensor(
                features[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            chunks.append(model(xb).detach().cpu().numpy())
    log_w = np.concatenate(chunks).astype(np.float64) / temperature
    log_w = np.clip(log_w, -max_log_weight, max_log_weight)
    log_w -= log_w.max()
    return log_w


def normalized_weights(log_w: np.ndarray) -> np.ndarray:
    w = np.exp(log_w)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full_like(w, 1.0 / len(w), dtype=np.float64)
    return w / total


def systematic_resample(
    weights: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Low-variance multinomial resampling."""
    positions = (rng.random() + np.arange(n_samples)) / n_samples
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side="right")


def subsample_reference(
    ref: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(ref) <= max_samples:
        return ref
    idx = rng.choice(len(ref), size=max_samples, replace=False)
    return ref[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-method", default="nsf",
                    help="Source suffix in flow_vs_quantile. Use 'nsf' for an "
                         "unsuffixed default PFN-NPE file.")
    ap.add_argument("--output-method", default=None,
                    help="Output suffix. Default: rank_dr for nsf input, "
                         "otherwise {input_method}_rank_dr.")
    ap.add_argument("--in-dir", type=Path, default=FVQ_DIR)
    ap.add_argument("--out-dir", type=Path, default=FVQ_DIR)
    ap.add_argument("--n-ref", type=int, default=None,
                    help="Reference observations to process. Default: all in source npz.")
    ap.add_argument("--n-output", type=int, default=None,
                    help="Corrected posterior samples per observation. Default: source count.")
    ap.add_argument("--n-ratio-samples", type=int, default=5000,
                    help="Maximum samples per class for ratio classifier.")
    ap.add_argument("--pit-mode", choices=["separate", "base", "pooled"],
                    default="separate",
                    help="'separate' removes each group's 1D marginals and "
                         "targets copula mismatch; 'base' uses the base "
                         "posterior CDF for both groups; 'pooled' uses pooled ranks.")
    ap.add_argument("--space", choices=["z", "u"], default="z",
                    help="Classifier feature space: Gaussianized PIT z or raw PIT u.")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Divide classifier logits by this before exponentiating.")
    ap.add_argument("--max-log-weight", type=float, default=8.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.max_log_weight <= 0:
        raise ValueError("--max-log-weight must be positive")
    if not 0 <= args.val_frac < 1:
        raise ValueError("--val-frac must be in [0, 1)")

    source_path, source_method = source_npz_path(
        args.task, args.seed, args.input_method, args.in_dir,
    )
    output_method = args.output_method
    if output_method is None:
        output_method = "rank_dr" if source_method == "nsf" else f"{source_method}_rank_dr"
    out_path = args.out_dir / f"{args.task}_s{args.seed}_{output_method}.npz"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path}")
        return

    torch.manual_seed(args.seed)
    src = dict(np.load(source_path, allow_pickle=True))
    base_all = np.asarray(src["flow_samples"], dtype=np.float32)
    n_ref = base_all.shape[0] if args.n_ref is None else min(args.n_ref, base_all.shape[0])
    n_output = base_all.shape[1] if args.n_output is None else args.n_output
    dim_theta = base_all.shape[2]

    task = get_task(args.task)
    x_ref = np.asarray(src.get("x_ref"), dtype=np.float32) if "x_ref" in src else np.stack([
        task.get_observation(num_observation=i + 1).numpy().reshape(-1)
        for i in range(n_ref)
    ]).astype(np.float32)
    theta_true = (
        np.asarray(src.get("theta_true"), dtype=np.float32)
        if "theta_true" in src
        else np.stack([
            task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
            for i in range(n_ref)
        ]).astype(np.float32)
    )
    x_ref = x_ref[:n_ref]
    theta_true = theta_true[:n_ref]

    print(f"task={args.task} seed={args.seed}")
    print(f"  source: {source_path}")
    print(f"  output: {out_path}")
    print(f"  source_samples={base_all.shape}, n_output={n_output}, dim_theta={dim_theta}")
    print(f"  pit_mode={args.pit_mode}, space={args.space}, method={output_method}")

    corrected = np.zeros((n_ref, n_output, dim_theta), dtype=np.float32)
    ratio_diag: dict[str, list[float]] = {
        "ratio_ess": [],
        "ratio_ess_frac": [],
        "ratio_max_weight": [],
        "ratio_unique_frac": [],
        "ratio_log_weight_std": [],
        "ratio_val_bce": [],
        "ratio_val_acc": [],
        "ratio_n_classifier_per_class": [],
    }

    for obs_idx in range(n_ref):
        obs_seed = args.seed * 1009 + obs_idx
        obs_rng = np.random.default_rng(obs_seed)
        base = np.asarray(base_all[obs_idx], dtype=np.float32)
        ref_full = task.get_reference_posterior_samples(
            num_observation=obs_idx + 1,
        ).numpy().astype(np.float32)
        n_ratio = min(args.n_ratio_samples, len(base), len(ref_full))
        if n_ratio < 2:
            raise ValueError(
                f"Need at least 2 samples per class for density-ratio training; "
                f"got n_ratio={n_ratio} for obs {obs_idx + 1}."
            )
        ref_ratio = subsample_reference(ref_full, n_ratio, obs_rng)
        if len(base) > n_ratio:
            base_idx = obs_rng.choice(len(base), size=n_ratio, replace=False)
        else:
            base_idx = np.arange(len(base))

        base_feat_all, ref_feat = pit_features(
            base,
            ref_ratio,
            pit_mode=args.pit_mode,
            space=args.space,
            eps=args.eps,
        )
        base_feat_ratio = base_feat_all[base_idx]

        model, train_diag = train_ratio_classifier(
            base_feat_ratio,
            ref_feat,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            val_frac=args.val_frac,
            seed=obs_seed,
        )
        log_w = classifier_log_weights(
            model,
            base_feat_all,
            temperature=args.temperature,
            max_log_weight=args.max_log_weight,
            batch_size=args.batch_size,
        )
        weights = normalized_weights(log_w)
        sample_idx = systematic_resample(weights, n_output, obs_rng)
        corrected[obs_idx] = base[sample_idx].astype(np.float32, copy=False)

        ess = 1.0 / float(np.sum(weights**2))
        unique_frac = len(np.unique(sample_idx)) / float(n_output)
        ratio_diag["ratio_ess"].append(ess)
        ratio_diag["ratio_ess_frac"].append(ess / len(weights))
        ratio_diag["ratio_max_weight"].append(float(weights.max()))
        ratio_diag["ratio_unique_frac"].append(unique_frac)
        ratio_diag["ratio_log_weight_std"].append(float(np.std(log_w)))
        ratio_diag["ratio_val_bce"].append(train_diag["val_bce"])
        ratio_diag["ratio_val_acc"].append(train_diag["val_acc"])
        ratio_diag["ratio_n_classifier_per_class"].append(
            train_diag["n_classifier_per_class"],
        )
        print(
            f"  obs {obs_idx + 1:02d}: n_ratio={n_ratio} "
            f"val_bce={train_diag['val_bce']:.3f} "
            f"val_acc={train_diag['val_acc']:.3f} "
            f"ESS={ess:.1f}/{len(weights)} max_w={weights.max():.3f} "
            f"unique={unique_frac:.3f}"
        )

    taus = np.asarray(src.get("taus", DEFAULT_TAUS), dtype=np.float32)
    flow_q = np.zeros((n_ref, len(taus), dim_theta), dtype=np.float32)
    emp_q = np.zeros_like(flow_q)
    for obs_idx in range(n_ref):
        flow_q[obs_idx] = np.quantile(corrected[obs_idx], taus, axis=0)
        ref = task.get_reference_posterior_samples(num_observation=obs_idx + 1).numpy()
        emp_q[obs_idx] = np.quantile(ref, taus, axis=0)
    flow_pinball = float(_pinball_np(theta_true, flow_q.transpose(1, 0, 2), taus).mean())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    replaced = {
        "flow_samples", "flow_q", "emp_q", "flow_pinball", "flow_type",
        "theta_true", "x_ref", "taus", "task", "seed",
    }
    save_kwargs = {k: v for k, v in src.items() if k not in replaced}
    save_kwargs.update(
        taus=taus,
        flow_q=flow_q,
        emp_q=emp_q,
        flow_samples=corrected,
        theta_true=theta_true,
        x_ref=x_ref,
        flow_pinball=flow_pinball,
        task=args.task,
        seed=args.seed,
        flow_type=output_method,
        source_flow_type=source_method,
        source_npz=str(source_path),
        ratio_pit_mode=args.pit_mode,
        ratio_space=args.space,
        ratio_temperature=args.temperature,
        ratio_max_log_weight=args.max_log_weight,
        ratio_n_output=n_output,
        **{k: np.asarray(v, dtype=np.float32) for k, v in ratio_diag.items()},
    )
    np.savez(str(out_path), **save_kwargs)

    print(f"\nflow pinball: {flow_pinball:.4f}")
    print(
        "mean diagnostics: "
        f"ESS={np.mean(ratio_diag['ratio_ess']):.1f}, "
        f"ESS_frac={np.mean(ratio_diag['ratio_ess_frac']):.3f}, "
        f"val_acc={np.nanmean(ratio_diag['ratio_val_acc']):.3f}"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
