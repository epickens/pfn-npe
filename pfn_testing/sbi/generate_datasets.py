"""Pre-generate simulation datasets for contrastive fine-tuning experiments.

Decouples simulation (CPU-bound) from training (GPU-bound) so that:
- Datasets are fixed and reproducible
- GPU jobs just load data, no simulator dependencies
- Crash recovery is trivial (skip existing files)

Usage:
    # Generate all single-task datasets (48 files)
    uv run python -m pfn_testing.sbi.generate_datasets

    # Also generate multi-task mixed datasets (60 files total)
    uv run python -m pfn_testing.sbi.generate_datasets --multi

    # Just one task/budget for testing
    uv run python -m pfn_testing.sbi.generate_datasets \
        --tasks two_moons_distractors --budgets 1000 --seeds 42
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pfn_testing.sbi.load_datasets import load_multi_task, load_single_task  # noqa: F401 (re-export)
from pfn_testing.sbi.sbibm_utils import AVAILABLE_TASKS, simulate


# ═══════════════════════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════════════════════

DISTRACTOR_TASKS = [
    "two_moons_distractors",
    "gaussian_mixture_distractors",
    "bernoulli_glm_distractors",
    "sir_distractors",
]

DEFAULT_BUDGETS = [1000, 3000, 10000, 30000]
DEFAULT_SEEDS = [42, 123, 7]
DEFAULT_N_VAL = 2000


# ═══════════════════════════════════════════════════════════════════════════════
# Single-task generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_single_task(
    task_name: str,
    budget: int,
    seed: int,
    n_val: int,
    data_dir: Path,
) -> bool:
    """Generate and save a single-task dataset.

    Returns True if generated, False if skipped.
    """
    out_dir = data_dir / task_name
    out_path = out_dir / f"n{budget}_s{seed}.npz"

    if out_path.exists():
        print(f"  [SKIP] {out_path} (already exists)")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = simulate(task_name, n_train=budget, n_val=n_val, seed=seed)
    elapsed = time.time() - t0

    np.savez(
        str(out_path),
        thetas_train=data["thetas_train"],
        xs_train=data["xs_train"],
        thetas_val=data["thetas_val"],
        xs_val=data["xs_val"],
        dim_theta=data["dim_theta"],
        dim_x=data["dim_x"],
        task_name=task_name,
        seed=seed,
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {out_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-task generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_multi_task(
    tasks: list[str],
    budget_per_task: int,
    seed: int,
    n_val: int,
    data_dir: Path,
) -> bool:
    """Generate and save a multi-task mixed dataset.

    Each task contributes budget_per_task training samples and n_val
    validation samples. Arrays are stored with per-task indexed keys
    since tasks have different theta/x dimensions.

    Returns True if generated, False if skipped.
    """
    out_dir = data_dir / "multi_distractor"
    out_path = out_dir / f"n{budget_per_task}_s{seed}.npz"

    if out_path.exists():
        print(f"  [SKIP] {out_path} (already exists)")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    save_kwargs: dict = {
        "task_names": np.array(tasks),
        "n_tasks": len(tasks),
        "seed": seed,
        "budget_per_task": budget_per_task,
    }

    dim_thetas = []
    dim_xs = []

    for i, task_name in enumerate(tasks):
        print(f"    Simulating {task_name}...")
        data = simulate(task_name, n_train=budget_per_task, n_val=n_val, seed=seed)

        save_kwargs[f"thetas_train_{i}"] = data["thetas_train"]
        save_kwargs[f"xs_train_{i}"] = data["xs_train"]
        save_kwargs[f"thetas_val_{i}"] = data["thetas_val"]
        save_kwargs[f"xs_val_{i}"] = data["xs_val"]
        dim_thetas.append(data["dim_theta"])
        dim_xs.append(data["dim_x"])

    save_kwargs["dim_thetas"] = np.array(dim_thetas)
    save_kwargs["dim_xs"] = np.array(dim_xs)

    elapsed = time.time() - t0
    np.savez(str(out_path), **save_kwargs)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {out_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-generate simulation datasets for SBI experiments",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=DISTRACTOR_TASKS,
        help=f"Tasks to generate (default: all 4 distractor tasks)",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS,
        help="Training set sizes (default: 1000 3000 10000 30000)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Random seeds (default: 42 123 7)",
    )
    parser.add_argument(
        "--n-val", type=int, default=DEFAULT_N_VAL,
        help="Validation set size per task (default: 2000)",
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="Also generate multi-task mixed datasets",
    )
    parser.add_argument(
        "--data-dir", type=str, default="pfn_testing/sbi/data",
        help="Output directory (default: pfn_testing/sbi/data)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Validate tasks
    for task in args.tasks:
        if task not in AVAILABLE_TASKS:
            parser.error(f"Unknown task: {task}. Choose from: {AVAILABLE_TASKS}")

    # Count total work
    n_single = len(args.tasks) * len(args.budgets) * len(args.seeds)
    n_multi = len(args.budgets) * len(args.seeds) if args.multi else 0
    n_total = n_single + n_multi

    print("=" * 60)
    print("Dataset Generation")
    print("=" * 60)
    print(f"  Tasks:   {args.tasks}")
    print(f"  Budgets: {args.budgets}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Val size: {args.n_val}")
    print(f"  Multi-task: {args.multi}")
    print(f"  Output: {data_dir}/")
    print(f"  Total files: {n_total} ({n_single} single + {n_multi} multi)")
    print()

    generated = 0
    skipped = 0
    failed = 0
    t_start = time.time()
    current = 0

    # Single-task datasets
    for task in args.tasks:
        for budget in args.budgets:
            for seed in args.seeds:
                current += 1
                print(f"[{current}/{n_total}] {task} n={budget} seed={seed}")
                try:
                    if generate_single_task(task, budget, seed, args.n_val, data_dir):
                        generated += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"  [FAIL] {e}")
                    failed += 1

    # Multi-task datasets
    if args.multi:
        for budget in args.budgets:
            for seed in args.seeds:
                current += 1
                print(f"[{current}/{n_total}] multi_distractor "
                      f"n={budget}/task seed={seed}")
                try:
                    if generate_multi_task(
                        args.tasks, budget, seed, args.n_val, data_dir,
                    ):
                        generated += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"  [FAIL] {e}")
                    failed += 1

    # Summary
    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("  GENERATION COMPLETE")
    print(f"  Generated: {generated}  Skipped: {skipped}  Failed: {failed}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  Output: {data_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
