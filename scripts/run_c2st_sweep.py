"""Run C2ST decomposition over every (task, seed, method) cell on disk.

Walks `flow_vs_quantile/` for posterior-sample npzs and writes matching
`c2st_decomp/` outputs by reusing `c2st_decomposition.per_obs_c2sts`
in-process (no subprocess overhead). Idempotent — skips cells whose
decomp file already exists.

Usage:
  uv run python scripts/run_c2st_sweep.py
  uv run python scripts/run_c2st_sweep.py --task slcp_distractors
  uv run python scripts/run_c2st_sweep.py --include-method copula
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task  # noqa: E402
from scripts.c2st_decomposition import per_obs_c2sts  # noqa: E402

FVQ_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")

NAME_RE = re.compile(r"(?P<task>.+?)_s(?P<seed>\d+)(?:_(?P<method>.+))?$")


def parse_name(stem: str) -> tuple[str, int, str] | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    return m.group("task"), int(m.group("seed")), m.group("method") or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None,
                    help="Restrict to a single task (e.g. slcp_distractors).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Restrict to a single seed.")
    ap.add_argument("--include-method", default=None,
                    help="Substring filter on method (e.g. 'copula', 'our_ar').")
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if c2st_decomp output exists.")
    args = ap.parse_args()

    DECOMP_DIR.mkdir(parents=True, exist_ok=True)
    task_cache: dict[str, object] = {}

    paths = sorted(FVQ_DIR.glob("*.npz"))
    print(f"Scanning {len(paths)} flow_vs_quantile npzs...")

    n_ran = n_skipped = n_bad = 0
    for fvq_path in paths:
        parsed = parse_name(fvq_path.stem)
        if not parsed:
            n_bad += 1
            continue
        task, seed, method = parsed
        if args.task and task != args.task:
            continue
        if args.seed is not None and seed != args.seed:
            continue
        if args.include_method and args.include_method not in method:
            continue

        suffix = f"_{method}" if method else ""
        out_path = DECOMP_DIR / f"{task}_s{seed}{suffix}.npz"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        try:
            data = np.load(fvq_path, allow_pickle=True)
            flow_samples = data["flow_samples"]
        except (KeyError, OSError) as e:
            print(f"  [bad-npz] {fvq_path.name}: {e}")
            n_bad += 1
            continue

        if task not in task_cache:
            task_cache[task] = get_task(task)
        sbibm_task = task_cache[task]

        n_ref = min(args.n_ref, flow_samples.shape[0])
        dim_theta = flow_samples.shape[2]
        joint = np.zeros(n_ref)
        marginal = np.zeros(n_ref)
        marginal_per_dim = np.zeros((n_ref, dim_theta))
        rank = np.zeros(n_ref)

        print(f"  [run] {task} s{seed} {method or 'nsf':<22}  "
              f"flow_samples={tuple(flow_samples.shape)}")
        for i in range(n_ref):
            mcmc = sbibm_task.get_reference_posterior_samples(
                num_observation=i + 1).numpy()
            res = per_obs_c2sts(flow_samples[i], mcmc)
            joint[i] = res["joint"]
            marginal[i] = res["marginal_mean"]
            marginal_per_dim[i] = res["marginal_per_dim"]
            rank[i] = res["rank"]

        np.savez(
            str(out_path),
            joint=joint, marginal=marginal, rank=rank,
            marginal_per_dim=marginal_per_dim,
            task=task, seed=seed, flow_type=method or "nsf",
        )
        print(f"     joint={joint.mean():.3f}±{joint.std():.3f}  "
              f"marg={marginal.mean():.3f}  rank={rank.mean():.3f}  "
              f"→ {out_path.name}")
        n_ran += 1

    print(f"\nDone. ran={n_ran} skipped={n_skipped} bad={n_bad}")


if __name__ == "__main__":
    main()
