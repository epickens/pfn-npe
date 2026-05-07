"""Populate plots/ folder in each per-layer sweep directory.

Reads the saved flow_tabpfn.pt, re-simulates training data (deterministic
from seed), extracts a TabPFN embedding for observation 1 at the
appropriate layer, and calls plot_diagnostics with the flow's posterior
samples vs the reference posterior samples.

The training-curve panel will be empty — run_layer_sweep doesn't save
training history. Posterior scatter + marginals come out correctly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (
    ProjectedFlow,  # noqa: F401  (keep for checkpoint loads if needed)
    build_flow,
    get_flow_defaults,
    sample_posterior,
)
import matplotlib.pyplot as plt

from pfn_testing.sbi.sbibm_utils import get_task, simulate
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder

N_LAYERS = 12
N_POSTERIOR_SAMPLES = 10_000


def plot_posterior_diagnostics(
    posterior_samples: np.ndarray,
    reference_samples: np.ndarray,
    label: str,
    output_path: str | Path,
    title_suffix: str = "",
) -> None:
    """Posterior scatter (first 2 θ dims) + up to 4 marginal histograms.

    Intentionally omits the training-curve panel that plot_diagnostics
    includes — run_layer_sweep does not save training history.
    """
    dim_theta = posterior_samples.shape[1]
    n_marginals = min(dim_theta, 4)
    n_panels = 1 + n_marginals

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    d0, d1 = 0, min(1, dim_theta - 1)
    ax.scatter(
        reference_samples[:2000, d0], reference_samples[:2000, d1],
        alpha=0.15, s=1, label="Reference", color="C0",
    )
    ax.scatter(
        posterior_samples[:2000, d0], posterior_samples[:2000, d1],
        alpha=0.15, s=1, label=label, color="C1",
    )
    ax.set_xlabel(rf"$\theta_{d0 + 1}$")
    ax.set_ylabel(rf"$\theta_{d1 + 1}$")
    ax.legend(markerscale=10)
    ax.set_title(f"Posterior samples{title_suffix}")

    for j in range(n_marginals):
        ax = axes[1 + j]
        ax.hist(reference_samples[:, j], bins=50, alpha=0.4,
                density=True, label="Reference", color="C0")
        ax.hist(posterior_samples[:, j], bins=50, alpha=0.4,
                density=True, label=label, color="C1")
        ax.set_xlabel(rf"$\theta_{j + 1}$")
        ax.set_title(f"Marginal {j + 1}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

TASKS = [
    "two_moons_distractors",
    "gaussian_mixture_distractors",
    "bernoulli_glm_distractors",
    "sir_distractors",
]


def run_task(task_name: str, n_train: int, seed: int, force: bool) -> None:
    base = Path(f"pfn_testing/sbi/outputs/{task_name}")
    # Check that layer dirs exist
    layer_dirs = [
        base / f"n{n_train}_per_dim_regressor_layer{k}_s{seed}"
        for k in range(N_LAYERS)
    ]
    missing = [d for d in layer_dirs if not (d / "flows" / "flow_tabpfn.pt").exists()]
    if missing:
        print(f"[{task_name}] missing flows for {len(missing)}/{N_LAYERS} layers — skipping.")
        return

    print(f"\n=== {task_name} (seed {seed}, n_train {n_train}) ===")
    data = simulate(task_name, n_train, 2000, seed)
    dim_theta, dim_x = data["dim_theta"], data["dim_x"]

    # Normalization stats from thetas_train (matches train_flow's own derivation)
    thetas_train = data["thetas_train"]
    theta_mean = thetas_train.mean(axis=0)
    theta_std = thetas_train.std(axis=0) + 1e-8

    # Observation 1
    task = get_task(task_name)
    x_obs_1 = task.get_observation(num_observation=1).numpy().squeeze(0)
    ref_1 = task.get_reference_posterior_samples(num_observation=1).numpy()

    # One embedder, reused across layers
    embedder = TabPFNEmbedder(
        context_size=1000, seed=seed, label_strategy="per_dim",
        layer=0, model_type="regressor",
    )
    embedder.fit(data["xs_train"], thetas=data["thetas_train"])

    defaults = get_flow_defaults(dim_theta)
    n_transforms = defaults["n_transforms"]
    hidden_features = defaults["hidden_features"]

    for k in range(N_LAYERS):
        layer_dir = layer_dirs[k]
        out_path = layer_dir / "plots" / "tabpfn_emb_diagnostics.png"
        if out_path.exists() and not force:
            print(f"  layer {k:2d}  exists, skipping ({out_path.name})")
            continue

        # Embedding for the observation at this layer
        embedder.layer = k
        obs_emb = embedder.transform(x_obs_1.reshape(1, -1))[0]
        emb_dim = obs_emb.shape[0]

        # Build flow matching the training architecture, load weights
        flow = build_flow(dim_theta, emb_dim, n_transforms, hidden_features,
                          n_bins=8, flow_type="nsf")
        state = torch.load(
            str(layer_dir / "flows" / "flow_tabpfn.pt"),
            map_location="cpu", weights_only=True,
        )
        flow.load_state_dict(state)

        # Sample posterior
        post = sample_posterior(flow, obs_emb, theta_mean, theta_std,
                                n_samples=N_POSTERIOR_SAMPLES)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_posterior_diagnostics(
            post, ref_1,
            label=f"TabPFN emb (layer {k})",
            output_path=str(out_path),
            title_suffix=f" — {task_name} layer {k}",
        )
        print(f"  layer {k:2d}  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if the PNG already exists.")
    args = ap.parse_args()

    for t in args.tasks:
        run_task(t, args.n_train, args.seed, args.force)


if __name__ == "__main__":
    main()
