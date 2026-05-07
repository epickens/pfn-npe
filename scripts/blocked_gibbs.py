"""Blocked learned Gibbs sampler over TabPFN-PCA64 conditional heads.

This is the blocked analogue of `scripts/learned_gibbs.py`. It trains one
conditional flow per block,

    q_B(theta_B | theta_-B, e(x)),

where e(x) is the frozen TabPFN embedding reduced to PCA64. The default block
set is all pairs. For two-dimensional tasks this is a single full-joint block;
for higher-dimensional tasks this performs pairwise blocked updates.

Outputs use the standard `flow_vs_quantile` npz schema.

Usage:
  uv run python scripts/blocked_gibbs.py --task two_moons --seed 42
  uv run python scripts/run_c2st_sweep.py --include-method blocked_gibbs --force
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    train_flow,
)
from pfn_testing.sbi.sbibm_utils import get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import PCAReducer, TabPFNEmbedder  # noqa: E402
from scripts.layer_quantile_probe import _pinball_np  # noqa: E402
from scripts.learned_gibbs import (  # noqa: E402
    DEFAULT_TAUS,
    OUT_DIR,
    GibbsTrainConfig,
    choose_output_samples,
    source_npz_path,
)


def parse_blocks(spec: str, dim_theta: int) -> list[tuple[int, ...]]:
    """Parse a block specification.

    Supported values:
      - `pairs`: all pair blocks, or the full block if D <= 2
      - `full`: one block with all dimensions
      - `singletons`: scalar Gibbs, mostly for parity checks
      - explicit semicolon-separated blocks, e.g. `0,1;2,3;3,4`
    """
    if spec == "pairs":
        if dim_theta <= 2:
            return [tuple(range(dim_theta))]
        return [tuple(p) for p in itertools.combinations(range(dim_theta), 2)]
    if spec == "full":
        return [tuple(range(dim_theta))]
    if spec == "singletons":
        return [(d,) for d in range(dim_theta)]

    blocks: list[tuple[int, ...]] = []
    for raw_block in spec.split(";"):
        block = tuple(int(x.strip()) for x in raw_block.split(",") if x.strip())
        if not block:
            continue
        if len(set(block)) != len(block):
            raise ValueError(f"Duplicate dimension in block {block}")
        if min(block) < 0 or max(block) >= dim_theta:
            raise ValueError(f"Block {block} outside theta dimension {dim_theta}")
        blocks.append(tuple(sorted(block)))
    if not blocks:
        raise ValueError(f"No valid blocks parsed from {spec!r}")
    return blocks


def block_context(
    embeddings: np.ndarray,
    theta: np.ndarray,
    block: tuple[int, ...],
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
) -> np.ndarray:
    """Build [embedding, standardized theta outside block] contexts."""
    dim_theta = theta.shape[1]
    mask = np.ones(dim_theta, dtype=bool)
    mask[list(block)] = False
    theta_z = (theta - theta_mean) / theta_std
    other = theta_z[:, mask]
    return np.concatenate([embeddings, other], axis=1).astype(np.float32, copy=False)


def train_block_heads(
    blocks: list[tuple[int, ...]],
    embeddings_train: np.ndarray,
    embeddings_val: np.ndarray,
    theta_train: np.ndarray,
    theta_val: np.ndarray,
    cfg: GibbsTrainConfig,
    n_transforms: int,
    n_bins: int,
) -> tuple[list[torch.nn.Module], list[dict], np.ndarray, np.ndarray]:
    """Train one conditional NSF per block."""
    theta_mean = theta_train.mean(axis=0)
    theta_std = theta_train.std(axis=0) + 1e-8
    flows: list[torch.nn.Module] = []
    histories: list[dict] = []

    for block_idx, block in enumerate(blocks):
        ctx_train = block_context(
            embeddings_train, theta_train, block, theta_mean, theta_std,
        )
        ctx_val = block_context(embeddings_val, theta_val, block, theta_mean, theta_std)
        target_train = theta_train[:, block]
        target_val = theta_val[:, block]
        print(
            f"\nTraining block head {block_idx + 1}/{len(blocks)} "
            f"B={block} (target_dim={len(block)}, context={ctx_train.shape[1]})..."
        )
        flow = build_flow(
            dim_theta=len(block),
            dim_context=ctx_train.shape[1],
            n_transforms=n_transforms,
            hidden_features=cfg.hidden_features,
            n_bins=n_bins,
            flow_type="nsf",
        )
        history = train_flow(
            flow,
            target_train,
            ctx_train,
            target_val,
            ctx_val,
            cfg,
        )
        flows.append(flow)
        histories.append(history)

    return flows, histories, theta_mean, theta_std


def sample_block_batch(
    flow: torch.nn.Module,
    context: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Draw one block sample for each context row."""
    flow.eval()
    device = next(flow.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(context), batch_size):
            ctx = torch.tensor(
                context[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            samples = flow(ctx).sample().detach().cpu().numpy()
            out.append(samples)
    values = np.concatenate(out, axis=0).astype(np.float32, copy=False)
    return values * target_std.astype(np.float32) + target_mean.astype(np.float32)


def blocked_gibbs_sample_observation(
    init: np.ndarray,
    embedding_obs: np.ndarray,
    blocks: list[tuple[int, ...]],
    flows: list[torch.nn.Module],
    histories: list[dict],
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    n_sweeps: int,
    burn_in: int,
    thin: int,
    update_order: str,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run blocked pseudo-Gibbs chains initialized from saved samples."""
    if n_sweeps < 1:
        raise ValueError("n_sweeps must be positive")
    if burn_in >= n_sweeps:
        raise ValueError("burn_in must be smaller than n_sweeps")
    if thin < 1:
        raise ValueError("thin must be positive")

    state = np.asarray(init, dtype=np.float32).copy()
    emb = np.repeat(embedding_obs.reshape(1, -1), state.shape[0], axis=0)
    kept: list[np.ndarray] = []
    mean_abs_updates = np.zeros((n_sweeps, len(blocks)), dtype=np.float32)

    for sweep in range(n_sweeps):
        if update_order == "random":
            order = rng.permutation(len(blocks))
        elif update_order == "fixed":
            order = np.arange(len(blocks))
        else:
            raise ValueError(f"Unknown update_order: {update_order}")

        for block_idx in order:
            block = blocks[int(block_idx)]
            old = state[:, block].copy()
            ctx = block_context(emb, state, block, theta_mean, theta_std)
            hist = histories[int(block_idx)]
            state[:, block] = sample_block_batch(
                flows[int(block_idx)],
                ctx,
                np.asarray(hist["theta_mean"], dtype=np.float32),
                np.asarray(hist["theta_std"], dtype=np.float32),
                batch_size,
            )
            mean_abs_updates[sweep, int(block_idx)] = float(
                np.mean(np.abs(state[:, block] - old)),
            )

        if sweep >= burn_in and ((sweep - burn_in) % thin == 0):
            kept.append(state.copy())

    if not kept:
        kept.append(state.copy())
    samples = np.concatenate(kept, axis=0).astype(np.float32, copy=False)
    return samples, mean_abs_updates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-method", default="raw_tabq_ps1000")
    parser.add_argument("--output-method", default=None)
    parser.add_argument("--in-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--blocks", default="pairs",
                        help="pairs, full, singletons, or explicit `0,1;2,3`.")
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--n-ref", type=int, default=10)
    parser.add_argument("--n-output", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--context-size", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--n-transforms", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--n-sweeps", type=int, default=50)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--update-order", choices=["random", "fixed"], default="random")
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_path, source_method = source_npz_path(
        args.task, args.seed, args.input_method, args.in_dir,
    )
    output_method = args.output_method or (
        "blocked_gibbs"
        if source_method == "raw_tabq_ps1000"
        else f"{source_method}_blocked_gibbs"
    )
    out_path = args.out_dir / f"{args.task}_s{args.seed}_{output_method}.npz"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    src = dict(np.load(source_path, allow_pickle=True))
    init_all = np.asarray(src["flow_samples"], dtype=np.float32)
    n_ref = min(args.n_ref, init_all.shape[0])
    n_output = init_all.shape[1] if args.n_output is None else args.n_output

    print(f"task={args.task} seed={args.seed}")
    print(f"  init source: {source_path}")
    print(f"  output:      {out_path}")
    print(f"  init_samples={init_all.shape}, n_ref={n_ref}, n_output={n_output}")
    print(
        f"  Blocked Gibbs: blocks={args.blocks}, sweeps={args.n_sweeps}, "
        f"burn_in={args.burn_in}, thin={args.thin}, order={args.update_order}"
    )

    print("\nSimulating train/val data...")
    data = simulate(args.task, args.n_train, args.n_val, args.seed)
    dim_theta = int(data["dim_theta"])
    blocks = parse_blocks(args.blocks, dim_theta)
    print(f"  parsed blocks: {blocks}")
    defaults = get_flow_defaults(max(len(block) for block in blocks))
    n_transforms = args.n_transforms or defaults["n_transforms"]

    print("\nExtracting TabPFN embeddings...")
    embedder = TabPFNEmbedder(
        context_size=args.context_size,
        seed=args.seed,
        label_strategy="per_dim",
        layer=None,
        model_type="regressor",
    )
    embedder.fit(data["xs_train"], thetas=data["thetas_train"])
    emb_train_raw = embedder.transform(data["xs_train"])
    emb_val_raw = embedder.transform(data["xs_val"])
    reducer = PCAReducer(args.embed_dim)
    emb_train = reducer.fit_transform(emb_train_raw).astype(np.float32, copy=False)
    emb_val = reducer.transform(emb_val_raw).astype(np.float32, copy=False)
    print(f"  reduced train embeddings: {emb_train.shape}")

    cfg = GibbsTrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_features=args.hidden,
    )
    flows, histories, theta_mean, theta_std = train_block_heads(
        blocks,
        emb_train,
        emb_val,
        data["thetas_train"].astype(np.float32, copy=False),
        data["thetas_val"].astype(np.float32, copy=False),
        cfg,
        n_transforms=n_transforms,
        n_bins=args.n_bins,
    )

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
    emb_ref = reducer.transform(embedder.transform(x_ref)).astype(np.float32, copy=False)

    flow_samples = np.zeros((n_ref, n_output, dim_theta), dtype=np.float32)
    update_summaries = np.zeros((n_ref, args.n_sweeps, len(blocks)), dtype=np.float32)
    kept_counts = np.zeros(n_ref, dtype=np.int32)
    print("\nRunning blocked Gibbs sampler...")
    for obs_idx in range(n_ref):
        obs_rng = np.random.default_rng(args.seed * 1009 + obs_idx)
        samples_all, updates = blocked_gibbs_sample_observation(
            init=init_all[obs_idx],
            embedding_obs=emb_ref[obs_idx],
            blocks=blocks,
            flows=flows,
            histories=histories,
            theta_mean=theta_mean,
            theta_std=theta_std,
            n_sweeps=args.n_sweeps,
            burn_in=args.burn_in,
            thin=args.thin,
            update_order=args.update_order,
            batch_size=args.sample_batch_size,
            rng=obs_rng,
        )
        flow_samples[obs_idx] = choose_output_samples(samples_all, n_output, obs_rng)
        update_summaries[obs_idx] = updates
        kept_counts[obs_idx] = len(samples_all)
        print(
            f"  obs {obs_idx + 1:02d}: kept={len(samples_all)} "
            f"mean_abs_final_block_update={updates[-5:].mean():.4f}"
        )

    taus = np.asarray(src.get("taus", DEFAULT_TAUS), dtype=np.float32)
    flow_q = np.zeros((n_ref, len(taus), dim_theta), dtype=np.float32)
    emp_q = np.zeros_like(flow_q)
    for obs_idx in range(n_ref):
        flow_q[obs_idx] = np.quantile(flow_samples[obs_idx], taus, axis=0)
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
        flow_samples=flow_samples,
        theta_true=theta_true,
        x_ref=x_ref,
        flow_pinball=flow_pinball,
        task=args.task,
        seed=args.seed,
        flow_type=output_method,
        source_flow_type=source_method,
        source_npz=str(source_path),
        blocked_gibbs_blocks=np.asarray([" ".join(map(str, b)) for b in blocks]),
        blocked_gibbs_embed_dim=args.embed_dim,
        blocked_gibbs_n_sweeps=args.n_sweeps,
        blocked_gibbs_burn_in=args.burn_in,
        blocked_gibbs_thin=args.thin,
        blocked_gibbs_kept_counts=kept_counts,
        blocked_gibbs_update_mean_abs=update_summaries,
        conditional_best_epochs=np.asarray(
            [h["best_epoch"] for h in histories],
            dtype=np.int32,
        ),
        conditional_theta_mean=theta_mean.astype(np.float32),
        conditional_theta_std=theta_std.astype(np.float32),
    )
    np.savez(str(out_path), **save_kwargs)

    print(f"\nflow pinball: {flow_pinball:.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
