"""Learned-summary NPE baseline with a fixed MLP summary net.

This is the reviewer-control baseline for PFN-NPE: train a generic learned
summary network jointly with an NSF posterior estimator under the same
simulation budget, seeds, and reference observations. The architecture is fixed
across tasks by default; it is not intended to be a hand-tuned task-specific
summary network.

Outputs match the layer_ablation/flow_vs_quantile schema so the existing C2ST
decomposition and budget plotting scripts consume the results unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from scripts.layer_quantile_probe import _pinball_np  # noqa: E402

OUT_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")


class MLPSummaryNet(torch.nn.Module):
    """Fixed generic summary network used as the sbi embedding_net."""

    def __init__(
        self,
        dim_x: int,
        *,
        summary_dim: int,
        hidden_features: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = dim_x
        for width in hidden_features:
            layers.append(torch.nn.Linear(in_dim, width))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            in_dim = width
        layers.append(torch.nn.Linear(in_dim, summary_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _suffix(args: argparse.Namespace) -> str:
    suffix = "learned_summary_npe"
    if args.n_train != 10_000:
        suffix = f"{suffix}_n{args.n_train}"
    if args.summary_dim != 64:
        suffix = f"{suffix}_d{args.summary_dim}"
    if args.hidden_features != [256, 256]:
        hidden = "x".join(str(x) for x in args.hidden_features)
        suffix = f"{suffix}_h{hidden}"
    if args.dropout != 0.0:
        suffix = f"{suffix}_drop{args.dropout:g}"
    return suffix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--n-flow-samples", type=int, default=1_000)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--summary-dim", type=int, default=64)
    ap.add_argument("--hidden-features", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num-transforms", type=int, default=5)
    ap.add_argument("--num-bins", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _suffix(args)
    out_npz = OUT_DIR / f"{args.task}_s{args.seed}_{suffix}.npz"
    if out_npz.exists():
        print(f"[skip] {out_npz}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"task={args.task} seed={args.seed} n_train={args.n_train} method={suffix}")

    t_train_start = time.perf_counter()
    print("Simulating training data...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = int(data["dim_theta"])
    dim_x = int(data["dim_x"])
    print(f"  dim_theta={dim_theta}, dim_x={dim_x}, device={device}")

    task = get_task(args.task)
    prior_dist = task.get_prior_dist()

    from sbi.inference import SNPE_C
    from sbi.neural_nets import posterior_nn
    from sbi.utils.user_input_checks import process_prior

    prior, *_ = process_prior(prior_dist)
    embedding_net = MLPSummaryNet(
        dim_x,
        summary_dim=args.summary_dim,
        hidden_features=args.hidden_features,
        dropout=args.dropout,
    )
    print(
        "  Summary net: "
        f"{dim_x} -> {' -> '.join(str(h) for h in args.hidden_features)} "
        f"-> {args.summary_dim}"
    )

    density_estimator = posterior_nn(
        model="nsf",
        embedding_net=embedding_net,
        hidden_features=128,
        num_transforms=args.num_transforms,
        num_bins=args.num_bins,
        z_score_x="none",
    )
    inference = SNPE_C(prior=prior, density_estimator=density_estimator)

    theta_t = torch.tensor(data["thetas_train"], dtype=torch.float32)
    x_t = torch.tensor(data["xs_train"], dtype=torch.float32)
    inference.append_simulations(theta_t, x_t)
    print(f"  attached {theta_t.shape[0]} simulations; training learned-summary NSF...")
    posterior_estimator = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_num_epochs=args.max_epochs,
        stop_after_epochs=args.patience,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(posterior_estimator)
    t_train_end = time.perf_counter()
    print(f"[TIMING] phase=train duration={t_train_end - t_train_start:.3f}")

    x_ref = np.stack(
        [
            task.get_observation(num_observation=i + 1).numpy().reshape(-1)
            for i in range(args.n_ref)
        ]
    )
    theta_true = np.stack(
        [
            task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
            for i in range(args.n_ref)
        ]
    )

    t_sample_start = time.perf_counter()
    print("\nSampling per ref obs...")
    flow_samples = np.zeros(
        (args.n_ref, args.n_flow_samples, dim_theta),
        dtype=np.float32,
    )
    for i in range(args.n_ref):
        x_obs_t = torch.tensor(x_ref[i], dtype=torch.float32)
        samples = posterior.sample(
            (args.n_flow_samples,),
            x=x_obs_t,
            show_progress_bars=False,
        )
        flow_samples[i] = samples.cpu().numpy().astype(np.float32)
        print(f"  obs {i + 1}: drew {flow_samples.shape[1]} samples")
    t_sample_end = time.perf_counter()
    print(f"[TIMING] phase=sample duration={t_sample_end - t_sample_start:.3f}")

    taus = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float32)
    flow_q = np.zeros((args.n_ref, len(taus), dim_theta), dtype=np.float32)
    emp_q = np.zeros_like(flow_q)
    for i in range(args.n_ref):
        flow_q[i] = np.quantile(flow_samples[i], taus, axis=0)
        ref = task.get_reference_posterior_samples(num_observation=i + 1).numpy()
        emp_q[i] = np.quantile(ref, taus, axis=0)
    flow_pinball = float(
        _pinball_np(theta_true, flow_q.transpose(1, 0, 2), taus).mean()
    )

    np.savez(
        str(out_npz),
        taus=taus,
        flow_q=flow_q,
        emp_q=emp_q,
        flow_samples=flow_samples,
        theta_true=theta_true,
        x_ref=x_ref,
        flow_pinball=flow_pinball,
        task=args.task,
        seed=args.seed,
        flow_type=suffix,
        summary_dim=args.summary_dim,
        hidden_features=np.asarray(args.hidden_features, dtype=np.int64),
    )
    print(f"\nflow pinball: {flow_pinball:.4f}")
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
