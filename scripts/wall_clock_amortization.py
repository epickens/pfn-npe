"""Amortized wall-clock curves for repeated posterior queries.

This timing harness separates one-time setup cost from repeated posterior-query
cost. PFN-NPE flow sizes are trained once per task/seed, then cumulative
sampling time is measured over increasing numbers of query observations.

Default figure:
  - left:  two_moons, d_theta=2
  - right: bernoulli_glm, d_theta=10

Example:
  uv run python scripts/wall_clock_amortization.py --dry-run
  uv run python scripts/wall_clock_amortization.py --from-csv path/to.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from npe_pfn import TabPFN_Based_NPE_PFN

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    sample_posterior,
    train_flow,
)
from pfn_testing.sbi.sbibm_utils import compute_c2st, get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import TabPFNEmbedder  # noqa: E402

OUT_DIR = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/wall_clock"
FIG_DIR = REPO_ROOT / "pfn_testing/sbi/outputs/layer_ablation/figures"

TASK_LABELS = {
    "two_moons": "Two moons",
    "bernoulli_glm": "Bernoulli GLM",
}
METHOD_LABELS = {
    "npe_pfn": "NPE-PFN",
    "pfn_npe_base": "PFN-NPE base",
    "pfn_npe_large": "PFN-NPE large",
    "pfn_npe_xl": "PFN-NPE XL",
}
METHOD_STYLES = {
    "npe_pfn": {"color": "#222222", "marker": "D"},
    "pfn_npe_base": {"color": "#0072B2", "marker": "o"},
    "pfn_npe_large": {"color": "#009E73", "marker": "^"},
    "pfn_npe_xl": {"color": "#CC79A7", "marker": "s"},
}

TASKS_DEFAULT = ["two_moons", "bernoulli_glm"]
SEEDS_DEFAULT = [1000, 1001, 1002]
QUERY_COUNTS_DEFAULT = [1, 2, 5, 10, 20, 50, 100]
FLOW_SIZES_DEFAULT = ["base", "large", "xl"]
TIMING_FIELDNAMES = [
    "task",
    "dim_theta",
    "seed",
    "method",
    "flow_size",
    "n_train",
    "context_size",
    "n_query",
    "n_flow_samples",
    "n_params",
    "simulate_s",
    "embed_s",
    "fit_s",
    "one_time_s",
    "sample_s",
    "total_s",
    "per_query_sample_s",
    "status",
    "error",
]
QUALITY_FIELDNAMES = [
    "task",
    "dim_theta",
    "seed",
    "method",
    "flow_size",
    "n_train",
    "context_size",
    "obs_idx",
    "n_flow_samples",
    "n_params",
    "sample_s",
    "joint_c2st",
    "status",
    "error",
]


@dataclass
class TrainConfig:
    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 5.0
    flow_type: str = "nsf"
    hidden_features: list[int] = field(default_factory=lambda: [128, 128])


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timer_start() -> float:
    sync_cuda()
    return time.perf_counter()


def timer_stop(t0: float) -> float:
    sync_cuda()
    return time.perf_counter() - t0


def flow_arch(dim_theta: int, size: str) -> tuple[int, list[int], int]:
    defaults = get_flow_defaults(dim_theta)
    if size == "base":
        return defaults["n_transforms"], defaults["hidden_features"], 8
    if size == "large":
        return 12, [512, 512], 16
    if size == "xl":
        return 20, [512, 512], 16
    raise ValueError(f"Unknown flow size: {size}")


def make_query_x(task_name: str, n_queries: int, seed: int) -> np.ndarray:
    """Simulate query observations from the task prior for timing only."""
    task = get_task(task_name)
    prior = task.get_prior()
    simulator = task.get_simulator()
    torch.manual_seed(seed)
    theta_q = prior(num_samples=n_queries)
    x_q = simulator(theta_q)
    return x_q.numpy().reshape(n_queries, -1)


def cumulative_query_times(
    sample_one,
    query_x: np.ndarray,
    query_counts: list[int],
) -> tuple[dict[int, float], str]:
    max_q = max(query_counts)
    cumulative: list[float] = []
    elapsed = 0.0
    for i in range(max_q):
        t0 = timer_start()
        try:
            sample_one(query_x[i])
        except Exception as exc:  # noqa: BLE001 - keep batch timing jobs alive.
            error = f"{type(exc).__name__}: {exc}"
            print(f"  [warn] sampling failed at query {i + 1}: {error}")
            return {
                q: cumulative[q - 1] for q in query_counts if q <= len(cumulative)
            }, error
        else:
            elapsed += timer_stop(t0)
        cumulative.append(elapsed)
    return {q: cumulative[q - 1] for q in query_counts}, ""


def evaluate_quality(
    sample_one,
    *,
    task_name: str,
    seed: int,
    method: str,
    flow_size: str,
    n_train: int,
    context_size: int,
    n_flow_samples: int,
    n_ref: int,
    n_params: int | None = None,
) -> list[dict[str, str]]:
    """Compute per-observation joint C2ST for standard sbibm observations."""
    if n_ref <= 0:
        return []

    task = get_task(task_name)
    rows = []
    for obs_idx in range(1, n_ref + 1):
        t0 = timer_start()
        try:
            x_obs = task.get_observation(num_observation=obs_idx).numpy().reshape(-1)
            samples = sample_one(x_obs)
            sample_s = timer_stop(t0)
            if not np.isfinite(samples).all():
                raise ValueError("posterior samples contain non-finite values")
            ref = task.get_reference_posterior_samples(num_observation=obs_idx).numpy()
            joint_c2st = compute_c2st(samples, ref)
        except Exception as exc:  # noqa: BLE001 - record quality failures per obs.
            sample_s = timer_stop(t0)
            error = f"{type(exc).__name__}: {exc}"
            print(f"  [warn] quality failed obs {obs_idx}: {error}")
            rows.append(
                {
                    "task": task_name,
                    "dim_theta": "",
                    "seed": str(seed),
                    "method": method,
                    "flow_size": flow_size,
                    "n_train": str(n_train),
                    "context_size": str(context_size),
                    "obs_idx": str(obs_idx),
                    "n_flow_samples": str(n_flow_samples),
                    "n_params": "" if n_params is None else str(n_params),
                    "sample_s": f"{sample_s:.3f}",
                    "joint_c2st": "",
                    "status": "failed",
                    "error": error,
                }
            )
            continue

        rows.append(
            {
                "task": task_name,
                "dim_theta": str(samples.shape[1]),
                "seed": str(seed),
                "method": method,
                "flow_size": flow_size,
                "n_train": str(n_train),
                "context_size": str(context_size),
                "obs_idx": str(obs_idx),
                "n_flow_samples": str(n_flow_samples),
                "n_params": "" if n_params is None else str(n_params),
                "sample_s": f"{sample_s:.3f}",
                "joint_c2st": f"{joint_c2st:.6f}",
                "status": "ok",
                "error": "",
            }
        )
    return rows


def run_npe_pfn(
    *,
    task_name: str,
    seed: int,
    data: dict,
    query_x: np.ndarray,
    query_counts: list[int],
    n_flow_samples: int,
    filter_context_size: int,
    simulate_s: float,
    quality_n_ref: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    prior_dist = get_task(task_name).get_prior_dist()
    theta_t = torch.tensor(data["thetas_train"], dtype=torch.float32)
    x_t = torch.tensor(data["xs_train"], dtype=torch.float32)

    t0 = timer_start()
    estimator = TabPFN_Based_NPE_PFN(
        prior=prior_dist,
        filter_context_size=filter_context_size,
        show_progress_bars=False,
    )
    estimator.append_simulations(theta_t, x_t)
    append_s = timer_stop(t0)

    def sample_one(x_obs: np.ndarray) -> np.ndarray:
        x_obs_t = torch.tensor(x_obs.reshape(1, -1), dtype=torch.float32)
        samples = estimator.sample(
            sample_shape=torch.Size([n_flow_samples]),
            x=x_obs_t,
        )
        return samples.cpu().numpy()

    sample_by_q, sample_error = cumulative_query_times(
        sample_one, query_x, query_counts
    )
    one_time_s = simulate_s + append_s
    rows = []
    for q in query_counts:
        sample_s = sample_by_q.get(q)
        status = "ok" if sample_s is not None else "failed"
        rows.append(
            {
                "task": task_name,
                "dim_theta": str(data["dim_theta"]),
                "seed": str(seed),
                "method": "npe_pfn",
                "flow_size": "",
                "n_train": str(len(data["thetas_train"])),
                "context_size": str(filter_context_size),
                "n_query": str(q),
                "n_flow_samples": str(n_flow_samples),
                "n_params": "",
                "simulate_s": f"{simulate_s:.3f}",
                "embed_s": "0.000",
                "fit_s": f"{append_s:.3f}",
                "one_time_s": f"{one_time_s:.3f}",
                "sample_s": "" if sample_s is None else f"{sample_s:.3f}",
                "total_s": "" if sample_s is None else f"{one_time_s + sample_s:.3f}",
                "per_query_sample_s": "" if sample_s is None else f"{sample_s / q:.3f}",
                "status": status,
                "error": sample_error if status == "failed" else "",
            }
        )
    quality_rows = evaluate_quality(
        sample_one,
        task_name=task_name,
        seed=seed,
        method="npe_pfn",
        flow_size="",
        n_train=len(data["thetas_train"]),
        context_size=filter_context_size,
        n_flow_samples=n_flow_samples,
        n_ref=quality_n_ref,
    )
    return rows, quality_rows


def run_pfn_npe_flow(
    *,
    task_name: str,
    seed: int,
    data: dict,
    query_x: np.ndarray,
    query_counts: list[int],
    n_flow_samples: int,
    context_size: int,
    max_epochs: int,
    lr: float,
    flow_size: str,
    simulate_s: float,
    embedder: TabPFNEmbedder,
    emb_train: np.ndarray,
    emb_val: np.ndarray,
    embed_s: float,
    quality_n_ref: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    n_transforms, hidden_features, n_bins = flow_arch(data["dim_theta"], flow_size)
    flow = build_flow(
        dim_theta=data["dim_theta"],
        dim_context=emb_train.shape[1],
        n_transforms=n_transforms,
        hidden_features=hidden_features,
        n_bins=n_bins,
    )
    n_params = sum(p.numel() for p in flow.parameters())
    cfg = TrainConfig(lr=lr, max_epochs=max_epochs)

    t0 = timer_start()
    history = train_flow(
        flow,
        data["thetas_train"],
        emb_train,
        data["thetas_val"],
        emb_val,
        cfg,
    )
    fit_s = timer_stop(t0)

    def sample_one(x_obs: np.ndarray) -> np.ndarray:
        e_obs = embedder.transform(x_obs.reshape(1, -1))[0]
        return sample_posterior(
            flow,
            e_obs,
            history["theta_mean"],
            history["theta_std"],
            n_flow_samples,
        )

    sample_by_q, sample_error = cumulative_query_times(
        sample_one, query_x, query_counts
    )
    one_time_s = simulate_s + embed_s + fit_s
    method = f"pfn_npe_{flow_size}"
    rows = []
    for q in query_counts:
        sample_s = sample_by_q.get(q)
        status = "ok" if sample_s is not None else "failed"
        rows.append(
            {
                "task": task_name,
                "dim_theta": str(data["dim_theta"]),
                "seed": str(seed),
                "method": method,
                "flow_size": flow_size,
                "n_train": str(len(data["thetas_train"])),
                "context_size": str(context_size),
                "n_query": str(q),
                "n_flow_samples": str(n_flow_samples),
                "n_params": str(n_params),
                "simulate_s": f"{simulate_s:.3f}",
                "embed_s": f"{embed_s:.3f}",
                "fit_s": f"{fit_s:.3f}",
                "one_time_s": f"{one_time_s:.3f}",
                "sample_s": "" if sample_s is None else f"{sample_s:.3f}",
                "total_s": "" if sample_s is None else f"{one_time_s + sample_s:.3f}",
                "per_query_sample_s": "" if sample_s is None else f"{sample_s / q:.3f}",
                "status": status,
                "error": sample_error if status == "failed" else "",
            }
        )
    quality_rows = evaluate_quality(
        sample_one,
        task_name=task_name,
        seed=seed,
        method=method,
        flow_size=flow_size,
        n_train=len(data["thetas_train"]),
        context_size=context_size,
        n_flow_samples=n_flow_samples,
        n_ref=quality_n_ref,
        n_params=n_params,
    )
    return rows, quality_rows


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


def write_quality_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUALITY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


def failed_rows(
    *,
    task_name: str,
    seed: int,
    data: dict,
    method: str,
    flow_size: str,
    n_train: int,
    context_size: int,
    query_counts: list[int],
    n_flow_samples: int,
    error: str,
) -> list[dict[str, str]]:
    return [
        {
            "task": task_name,
            "dim_theta": str(data.get("dim_theta", "")),
            "seed": str(seed),
            "method": method,
            "flow_size": flow_size,
            "n_train": str(n_train),
            "context_size": str(context_size),
            "n_query": str(q),
            "n_flow_samples": str(n_flow_samples),
            "n_params": "",
            "simulate_s": "",
            "embed_s": "",
            "fit_s": "",
            "one_time_s": "",
            "sample_s": "",
            "total_s": "",
            "per_query_sample_s": "",
            "status": "failed",
            "error": error,
        }
        for q in query_counts
    ]


def failed_quality_rows(
    *,
    task_name: str,
    seed: int,
    data: dict,
    method: str,
    flow_size: str,
    n_train: int,
    context_size: int,
    n_ref: int,
    n_flow_samples: int,
    error: str,
) -> list[dict[str, str]]:
    return [
        {
            "task": task_name,
            "dim_theta": str(data.get("dim_theta", "")),
            "seed": str(seed),
            "method": method,
            "flow_size": flow_size,
            "n_train": str(n_train),
            "context_size": str(context_size),
            "obs_idx": str(obs_idx),
            "n_flow_samples": str(n_flow_samples),
            "n_params": "",
            "sample_s": "",
            "joint_c2st": "",
            "status": "failed",
            "error": error,
        }
        for obs_idx in range(1, n_ref + 1)
    ]


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def aggregate(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[int, list[float]]]]:
    out: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        total_s = row.get("total_s", "")
        if not total_s:
            continue
        out[row["task"]][row["method"]][int(row["n_query"])].append(float(total_s))
    return out


def plot_amortization(
    rows: list[dict[str, str]],
    *,
    tasks: list[str],
    methods: list[str],
    out: Path,
    errorbar: str,
) -> None:
    import matplotlib.pyplot as plt

    by_task = aggregate(rows)
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(3.5 * len(tasks), 2.8), sharey=True
    )
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks, strict=False):
        for method in methods:
            by_q = by_task.get(task, {}).get(method, {})
            q_values = sorted(by_q)
            if not q_values:
                continue
            means = [float(np.mean(by_q[q])) for q in q_values]
            yerr = None
            if errorbar != "none":
                spreads = []
                for q in q_values:
                    values = by_q[q]
                    if len(values) <= 1:
                        spreads.append(0.0)
                        continue
                    spread = float(np.std(values))
                    if errorbar == "sem":
                        spread /= float(np.sqrt(len(values)))
                    spreads.append(spread)
                yerr = spreads
            ax.errorbar(
                q_values,
                means,
                yerr=yerr,
                label=METHOD_LABELS.get(method, method),
                linewidth=2.2,
                markersize=4.5,
                capsize=2.5 if yerr is not None else 0,
                **METHOD_STYLES.get(method, {}),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TASK_LABELS.get(task, task.replace("_", " ")), fontsize=10)
        ax.set_xlabel("Posterior queries")
        ax.grid(True, alpha=0.2)
        dim_values = sorted(
            {
                row["dim_theta"]
                for row in rows
                if row["task"] == task and row.get("dim_theta")
            }
        )
        if dim_values:
            ax.text(
                0.04,
                0.92,
                f"d_theta={dim_values[0]}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )

    axes[0].set_ylabel("Total wall-clock time (s)")
    handles, _ = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}\nWrote {pdf_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument("--query-counts", type=int, nargs="+", default=QUERY_COUNTS_DEFAULT)
    ap.add_argument(
        "--flow-sizes",
        nargs="+",
        default=FLOW_SIZES_DEFAULT,
        choices=["base", "large", "xl"],
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["npe_pfn", "pfn_npe"],
        choices=["npe_pfn", "pfn_npe"],
    )
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-flow-samples", type=int, default=1000)
    ap.add_argument("--context-size", type=int, default=10_000)
    ap.add_argument("--filter-context-size", type=int, default=10_000)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument(
        "--csv-out", type=Path, default=OUT_DIR / "wall_clock_amortization.csv"
    )
    ap.add_argument(
        "--quality-out",
        type=Path,
        default=OUT_DIR / "wall_clock_amortization_quality.csv",
    )
    ap.add_argument(
        "--quality-n-ref",
        type=int,
        default=0,
        help="If >0, compute joint C2ST on this many standard reference observations.",
    )
    ap.add_argument(
        "--plot-out", type=Path, default=FIG_DIR / "wall_clock_amortization.png"
    )
    ap.add_argument("--from-csv", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--errorbar", choices=["std", "sem", "none"], default="std")
    args = ap.parse_args()

    plot_methods = ["npe_pfn"] + [f"pfn_npe_{s}" for s in args.flow_sizes]
    if args.from_csv is not None:
        rows = read_rows(args.from_csv)
        plot_amortization(
            rows,
            tasks=args.tasks,
            methods=plot_methods,
            out=args.plot_out,
            errorbar=args.errorbar,
        )
        return

    if args.dry_run:
        for task in args.tasks:
            for seed in args.seeds:
                if "npe_pfn" in args.methods:
                    print(f"[dry-run] task={task} seed={seed} method=npe_pfn")
                if "pfn_npe" in args.methods:
                    for size in args.flow_sizes:
                        print(
                            f"[dry-run] task={task} seed={seed} method=pfn_npe size={size}"
                        )
        return

    rows: list[dict[str, str]] = []
    quality_rows: list[dict[str, str]] = []
    for task_name in args.tasks:
        for seed in args.seeds:
            print(f"\n=== task={task_name} seed={seed} ===")
            torch.manual_seed(seed)
            np.random.seed(seed)

            t0 = timer_start()
            data = simulate(task_name, args.n_train, args.n_val, seed)
            simulate_s = timer_stop(t0)
            query_x = make_query_x(task_name, max(args.query_counts), seed + 50_000)

            if "npe_pfn" in args.methods:
                print("\n[NPE-PFN]")
                try:
                    new_rows, new_quality_rows = run_npe_pfn(
                        task_name=task_name,
                        seed=seed,
                        data=data,
                        query_x=query_x,
                        query_counts=args.query_counts,
                        n_flow_samples=args.n_flow_samples,
                        filter_context_size=args.filter_context_size,
                        simulate_s=simulate_s,
                        quality_n_ref=args.quality_n_ref,
                    )
                except Exception as exc:  # noqa: BLE001 - record and continue batch job.
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"  [warn] NPE-PFN setup failed: {error}")
                    new_rows = failed_rows(
                        task_name=task_name,
                        seed=seed,
                        data=data,
                        method="npe_pfn",
                        flow_size="",
                        n_train=args.n_train,
                        context_size=args.filter_context_size,
                        query_counts=args.query_counts,
                        n_flow_samples=args.n_flow_samples,
                        error=error,
                    )
                    new_quality_rows = failed_quality_rows(
                        task_name=task_name,
                        seed=seed,
                        data=data,
                        method="npe_pfn",
                        flow_size="",
                        n_train=args.n_train,
                        context_size=args.filter_context_size,
                        n_ref=args.quality_n_ref,
                        n_flow_samples=args.n_flow_samples,
                        error=error,
                    )
                rows.extend(new_rows)
                write_rows(args.csv_out, rows)
                quality_rows.extend(new_quality_rows)
                if args.quality_n_ref > 0:
                    write_quality_rows(args.quality_out, quality_rows)

            if "pfn_npe" in args.methods:
                print("\n[PFN-NPE embedding]")
                t0 = timer_start()
                embedder = TabPFNEmbedder(
                    context_size=args.context_size,
                    seed=seed,
                    label_strategy="per_dim",
                    layer=None,
                    model_type="regressor",
                )
                embedder.fit(data["xs_train"], thetas=data["thetas_train"])
                emb_train = embedder.transform(data["xs_train"])
                emb_val = embedder.transform(data["xs_val"])
                embed_s = timer_stop(t0)

                for flow_size in args.flow_sizes:
                    print(f"\n[PFN-NPE size={flow_size}]")
                    try:
                        new_rows, new_quality_rows = run_pfn_npe_flow(
                            task_name=task_name,
                            seed=seed,
                            data=data,
                            query_x=query_x,
                            query_counts=args.query_counts,
                            n_flow_samples=args.n_flow_samples,
                            context_size=args.context_size,
                            max_epochs=args.max_epochs,
                            lr=args.lr,
                            flow_size=flow_size,
                            simulate_s=simulate_s,
                            embedder=embedder,
                            emb_train=emb_train,
                            emb_val=emb_val,
                            embed_s=embed_s,
                            quality_n_ref=args.quality_n_ref,
                        )
                    except Exception as exc:  # noqa: BLE001 - record and continue batch job.
                        error = f"{type(exc).__name__}: {exc}"
                        print(f"  [warn] PFN-NPE size={flow_size} failed: {error}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        new_rows = failed_rows(
                            task_name=task_name,
                            seed=seed,
                            data=data,
                            method=f"pfn_npe_{flow_size}",
                            flow_size=flow_size,
                            n_train=args.n_train,
                            context_size=args.context_size,
                            query_counts=args.query_counts,
                            n_flow_samples=args.n_flow_samples,
                            error=error,
                        )
                        new_quality_rows = failed_quality_rows(
                            task_name=task_name,
                            seed=seed,
                            data=data,
                            method=f"pfn_npe_{flow_size}",
                            flow_size=flow_size,
                            n_train=args.n_train,
                            context_size=args.context_size,
                            n_ref=args.quality_n_ref,
                            n_flow_samples=args.n_flow_samples,
                            error=error,
                        )
                    rows.extend(new_rows)
                    write_rows(args.csv_out, rows)
                    quality_rows.extend(new_quality_rows)
                    if args.quality_n_ref > 0:
                        write_quality_rows(args.quality_out, quality_rows)

    write_rows(args.csv_out, rows)
    if args.quality_n_ref > 0:
        write_quality_rows(args.quality_out, quality_rows)
    plot_amortization(
        rows,
        tasks=args.tasks,
        methods=plot_methods,
        out=args.plot_out,
        errorbar=args.errorbar,
    )


if __name__ == "__main__":
    main()
