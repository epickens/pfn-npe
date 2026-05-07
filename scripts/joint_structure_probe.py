"""Probe whether PFN-NPE loses joint posterior structure before or inside the flow.

The key diagnostic uses simulation pairs directly. For squared-error regression,

    f_1(x) = argmin E ||f(x) - theta||^2        = E[theta | x]
    f_2(x) = argmin E ||f(x) - theta theta^T||^2 = E[theta theta^T | x]

so a simple probe trained on (summary(x), theta, theta theta^T) estimates
posterior first and second moments. If a probe on TabPFN summaries can recover
reference posterior correlations but the conditional flow cannot sample them,
the bottleneck is likely the flow/objective. If raw x can recover them but
TabPFN summaries cannot, the bottleneck is likely representation loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    sample_posterior,
    train_flow,
)
from pfn_testing.sbi.sbibm_utils import compute_c2st, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import Config, PCAReducer, TabPFNEmbedder  # noqa: E402


def upper_indices(dim: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(dim)


def make_moment_targets(theta: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return [theta, upper_triangle(theta theta^T)] regression targets."""
    theta = np.asarray(theta, dtype=np.float32)
    ii, jj = upper_indices(theta.shape[1])
    second = theta[:, ii] * theta[:, jj]
    return np.concatenate([theta, second], axis=1), (ii, jj)


def rank_columns(x: np.ndarray) -> np.ndarray:
    """Column-wise empirical ranks in (0, 1)."""
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float64)
    denom = x.shape[0] + 1.0
    for d in range(x.shape[1]):
        order = np.argsort(x[:, d], kind="mergesort")
        out[order, d] = (np.arange(x.shape[0]) + 1.0) / denom
    return out


def corrcoef_safe(samples: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(samples, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def moments_from_samples(samples: np.ndarray) -> dict[str, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    corr = corrcoef_safe(samples)
    rank_corr = corrcoef_safe(rank_columns(samples))
    return {
        "mean": mean,
        "cov": cov,
        "corr": corr,
        "rank_corr": rank_corr,
    }


def moments_from_prediction(
    pred: np.ndarray,
    dim: int,
    tri: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert predicted [E theta, E theta theta^T] into covariance/correlation."""
    mean = pred[:dim].astype(np.float64)
    second_flat = pred[dim:].astype(np.float64)
    ii, jj = tri
    second = np.zeros((dim, dim), dtype=np.float64)
    second[ii, jj] = second_flat
    second[jj, ii] = second_flat
    cov = second - np.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)
    var = np.clip(np.diag(cov), 1e-8, None)
    denom = np.sqrt(np.outer(var, var))
    corr = np.clip(cov / denom, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return {"mean": mean, "cov": cov, "corr": corr}


def offdiag(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def moment_error_row(
    obs_id: int,
    method: str,
    kind: str,
    pred: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
) -> dict[str, Any]:
    cov_denom = np.linalg.norm(ref["cov"], ord="fro") + 1e-8
    corr_pred = offdiag(pred["corr"])
    corr_ref = offdiag(ref["corr"])
    row: dict[str, Any] = {
        "obs": obs_id,
        "method": method,
        "kind": kind,
        "mean_rmse": float(np.sqrt(np.mean((pred["mean"] - ref["mean"]) ** 2))),
        "cov_rel_fro": float(
            np.linalg.norm(pred["cov"] - ref["cov"], ord="fro") / cov_denom
        ),
        "corr_offdiag_mae": float(np.mean(np.abs(corr_pred - corr_ref))),
        "corr_offdiag_rmse": float(np.sqrt(np.mean((corr_pred - corr_ref) ** 2))),
        "corr_sign_acc": float(np.mean(np.sign(corr_pred) == np.sign(corr_ref))),
    }
    if "rank_corr" in pred and "rank_corr" in ref:
        row["rank_corr_offdiag_mae"] = float(
            np.mean(np.abs(offdiag(pred["rank_corr"]) - offdiag(ref["rank_corr"])))
        )
    return row


def pooled_rank_transform(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    pooled = np.concatenate([a, b], axis=0)
    ranked = rank_columns(pooled)
    return ranked[:n], ranked[n:]


def c2st_decomposition(
    samples: np.ndarray,
    ref: np.ndarray,
    folds: int,
    max_epochs: int,
    seed: int,
) -> dict[str, float]:
    n = min(len(samples), len(ref))
    a = samples[:n]
    b = ref[:n]
    joint = compute_c2st(a, b, n_folds=folds, max_epochs=max_epochs, seed=seed)
    marginal = float(
        np.mean(
            [
                compute_c2st(
                    a[:, d : d + 1],
                    b[:, d : d + 1],
                    n_folds=folds,
                    max_epochs=max_epochs,
                    seed=seed,
                )
                for d in range(a.shape[1])
            ]
        )
    )
    ar, br = pooled_rank_transform(a, b)
    rank = compute_c2st(ar, br, n_folds=folds, max_epochs=max_epochs, seed=seed)
    return {"joint_c2st": joint, "marginal_c2st": marginal, "rank_c2st": rank}


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


class MomentMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        last = dim_in
        for _ in range(depth):
            layers.extend([nn.Linear(last, hidden), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = hidden
        layers.append(nn.Linear(last, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrainedMomentProbe:
    def __init__(
        self,
        model: MomentMLP,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        device: torch.device,
    ):
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        xz = standardize_apply(x, self.x_mean, self.x_std)
        self.model.eval()
        with torch.no_grad():
            pred_z = self.model(
                torch.tensor(xz, dtype=torch.float32, device=self.device)
            ).cpu().numpy()
        return pred_z * self.y_std + self.y_mean


def train_moment_probe(
    name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[TrainedMomentProbe, dict[str, Any]]:
    print(f"\nTraining moment probe: {name}  x_dim={x_train.shape[1]}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_train_z, x_mean, x_std = standardize_fit(x_train.astype(np.float32))
    x_val_z = standardize_apply(x_val.astype(np.float32), x_mean, x_std)
    y_train_z, y_mean, y_std = standardize_fit(y_train.astype(np.float32))
    y_val_z = standardize_apply(y_val.astype(np.float32), y_mean, y_std)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = MomentMLP(
        x_train.shape[1],
        y_train.shape[1],
        hidden=args.probe_hidden,
        depth=args.probe_depth,
        dropout=args.probe_dropout,
    ).to(device)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(x_train_z, dtype=torch.float32),
        torch.tensor(y_train_z, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.probe_batch_size, shuffle=True
    )
    x_val_t = torch.tensor(x_val_z, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_z, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.probe_epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = ((model(x_val_t) - y_val_t) ** 2).mean().item()
        train_loss = float(np.mean(losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if epoch % 25 == 0:
            print(f"  epoch {epoch:4d}: train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.probe_patience:
            print(f"  early stopping at epoch {epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    return (
        TrainedMomentProbe(model, x_mean, x_std, y_mean, y_std, device),
        {
            "best_val_loss": best_val,
            "best_epoch": int(np.argmin(history["val_loss"])),
            "history": history,
        },
    )


def train_context_flow(
    name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    theta_train: np.ndarray,
    theta_val: np.ndarray,
    cfg: Config,
) -> tuple[nn.Module, dict[str, Any]]:
    dim_theta = theta_train.shape[1]
    defaults = get_flow_defaults(dim_theta)
    n_transforms = cfg.n_transforms or defaults["n_transforms"]
    hidden_features = cfg.hidden_features or defaults["hidden_features"]
    print(
        f"\nTraining flow: {name}  context_dim={x_train.shape[1]} "
        f"transforms={n_transforms} hidden={hidden_features}"
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    flow = build_flow(
        dim_theta=dim_theta,
        dim_context=x_train.shape[1],
        n_transforms=n_transforms,
        hidden_features=hidden_features,
        n_bins=cfg.n_bins,
        flow_type=cfg.flow_type,
    )
    history = train_flow(flow, theta_train, x_train, theta_val, x_val, cfg)
    return flow, history


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(r["method"], r["kind"]) for r in rows})
    metrics = [
        "mean_rmse",
        "cov_rel_fro",
        "corr_offdiag_mae",
        "corr_offdiag_rmse",
        "corr_sign_acc",
        "rank_corr_offdiag_mae",
        "joint_c2st",
        "marginal_c2st",
        "rank_c2st",
    ]
    out = []
    for method, kind in keys:
        subset = [r for r in rows if r["method"] == method and r["kind"] == kind]
        row: dict[str, Any] = {"method": method, "kind": kind, "n_obs": len(subset)}
        for metric in metrics:
            vals = [
                float(r[metric])
                for r in subset
                if metric in r and r[metric] is not None and not np.isnan(float(r[metric]))
            ]
            if vals:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="slcp")
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--context-size", type=int, default=1_000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--label-strategy", default="per_dim", choices=["per_dim", "per_dim_mean"])
    ap.add_argument("--model-type", default="regressor", choices=["regressor", "classifier"])
    ap.add_argument("--model-version", default="v2", choices=["v2", "v2.5"])
    ap.add_argument("--n-reference-observations", type=int, default=10)
    ap.add_argument("--n-posterior-samples", type=int, default=1_000)
    ap.add_argument("--probe-hidden", type=int, default=256)
    ap.add_argument("--probe-depth", type=int, default=2)
    ap.add_argument("--probe-dropout", type=float, default=0.05)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-weight-decay", type=float, default=1e-4)
    ap.add_argument("--probe-batch-size", type=int, default=256)
    ap.add_argument("--probe-epochs", type=int, default=250)
    ap.add_argument("--probe-patience", type=int, default=30)
    ap.add_argument("--flow-type", default="nsf", choices=["nsf", "naf"])
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--run-raw-flow", action="store_true")
    ap.add_argument("--run-c2st", action="store_true")
    ap.add_argument("--c2st-folds", type=int, default=3)
    ap.add_argument("--c2st-max-epochs", type=int, default=100)
    ap.add_argument("--out-dir", default="pfn_testing/sbi/outputs/joint_structure_probe")
    args = ap.parse_args()

    print(
        f"Simulating {args.task}: n_train={args.n_train}, n_val={args.n_val}, "
        f"seed={args.seed}"
    )
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    task = data["task"]
    dim_theta = data["dim_theta"]
    tri = upper_indices(dim_theta)
    y_train, _ = make_moment_targets(data["thetas_train"])
    y_val, _ = make_moment_targets(data["thetas_val"])

    obs_x = np.stack(
        [
            task.get_observation(num_observation=i).numpy().squeeze(0)
            for i in range(1, args.n_reference_observations + 1)
        ],
        axis=0,
    ).astype(np.float32)

    print("\nExtracting TabPFN summaries")
    embedder = TabPFNEmbedder(
        context_size=args.context_size,
        seed=args.seed,
        label_strategy=args.label_strategy,
        model_type=args.model_type,
        device=args.device,
        model_version=args.model_version,
    )
    embedder.fit(data["xs_train"], thetas=data["thetas_train"])
    emb_train = embedder.transform(data["xs_train"])
    emb_val = embedder.transform(data["xs_val"])
    emb_obs = embedder.transform(obs_x)
    print(f"  full embedding shapes: train={emb_train.shape} val={emb_val.shape} obs={emb_obs.shape}")

    pca = PCAReducer(args.embed_dim)
    emb_train_pca = pca.fit_transform(emb_train)
    emb_val_pca = pca.transform(emb_val)
    emb_obs_pca = pca.transform(emb_obs)

    contexts = {
        "raw_x": (data["xs_train"], data["xs_val"], obs_x),
        "tabpfn_full": (emb_train, emb_val, emb_obs),
        f"tabpfn_pca{args.embed_dim}": (emb_train_pca, emb_val_pca, emb_obs_pca),
    }

    probe_rows: list[dict[str, Any]] = []
    probe_histories: dict[str, Any] = {}
    ref_samples_by_obs: dict[int, np.ndarray] = {}
    ref_moments_by_obs: dict[int, dict[str, np.ndarray]] = {}
    for obs_id in range(1, args.n_reference_observations + 1):
        ref = task.get_reference_posterior_samples(num_observation=obs_id).numpy()
        ref = ref[: args.n_posterior_samples]
        ref_samples_by_obs[obs_id] = ref
        ref_moments_by_obs[obs_id] = moments_from_samples(ref)

    for name, (x_train, x_val, x_obs) in contexts.items():
        probe, history = train_moment_probe(name, x_train, x_val, y_train, y_val, args)
        probe_histories[name] = history
        pred_all = probe.predict(x_obs)
        for obs_idx, pred in enumerate(pred_all, start=1):
            pred_moments = moments_from_prediction(pred, dim_theta, tri)
            probe_rows.append(
                moment_error_row(
                    obs_idx,
                    name,
                    "moment_probe",
                    pred_moments,
                    ref_moments_by_obs[obs_idx],
                )
            )

    cfg = Config(
        task_name=args.task,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        device=args.device,
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        label_strategy=args.label_strategy,
        model_type=args.model_type,
        flow_type=args.flow_type,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        n_posterior_samples=args.n_posterior_samples,
        n_reference_observations=args.n_reference_observations,
        skip_raw=True,
    )
    flow_context_names = [f"tabpfn_pca{args.embed_dim}"]
    if args.run_raw_flow:
        flow_context_names.insert(0, "raw_x")

    for name in flow_context_names:
        x_train, x_val, x_obs = contexts[name]
        flow, history = train_context_flow(
            name,
            x_train,
            x_val,
            data["thetas_train"],
            data["thetas_val"],
            cfg,
        )
        for obs_idx in range(1, args.n_reference_observations + 1):
            samples = sample_posterior(
                flow,
                x_obs[obs_idx - 1],
                history["theta_mean"],
                history["theta_std"],
                args.n_posterior_samples,
            )
            pred = moments_from_samples(samples)
            row = moment_error_row(
                obs_idx,
                name,
                "flow_samples",
                pred,
                ref_moments_by_obs[obs_idx],
            )
            if args.run_c2st:
                row.update(
                    c2st_decomposition(
                        samples,
                        ref_samples_by_obs[obs_idx],
                        folds=args.c2st_folds,
                        max_epochs=args.c2st_max_epochs,
                        seed=args.seed,
                    )
                )
            probe_rows.append(row)

    summary = group_summary(probe_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.task}_n{args.n_train}_s{args.seed}_"
        f"{args.label_strategy}_{args.model_type}_pca{args.embed_dim}"
    )
    csv_path = out_dir / f"{stem}.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    json_path = out_dir / f"{stem}.json"

    fieldnames = sorted({k for row in probe_rows for k in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(probe_rows)

    summary_fieldnames = sorted({k for row in summary for k in row})
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    json_payload = {
        "config": {
            **asdict(cfg),
            "model_version": args.model_version,
            "probe_hidden": args.probe_hidden,
            "probe_depth": args.probe_depth,
            "run_raw_flow": args.run_raw_flow,
            "run_c2st": args.run_c2st,
        },
        "probe_histories": {
            k: {
                "best_val_loss": v["best_val_loss"],
                "best_epoch": v["best_epoch"],
            }
            for k, v in probe_histories.items()
        },
        "summary": summary,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, default=str)

    print("\nSummary")
    print("-" * 96)
    print(
        f"{'method':<16} {'kind':<13} {'corr_mae':>9} {'cov_fro':>9} "
        f"{'mean_rmse':>10} {'rank_mae':>9} {'joint':>8} {'rank':>8}"
    )
    for row in summary:
        print(
            f"{row['method']:<16} {row['kind']:<13} "
            f"{row.get('corr_offdiag_mae_mean', float('nan')):>9.4f} "
            f"{row.get('cov_rel_fro_mean', float('nan')):>9.4f} "
            f"{row.get('mean_rmse_mean', float('nan')):>10.4f} "
            f"{row.get('rank_corr_offdiag_mae_mean', float('nan')):>9.4f} "
            f"{row.get('joint_c2st_mean', float('nan')):>8.4f} "
            f"{row.get('rank_c2st_mean', float('nan')):>8.4f}"
        )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
