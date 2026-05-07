"""Compare PFN-embedding and raw-x quantile validation results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PFN_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/quantile_validate")
RAW_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/quantile_validate_raw")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")

TASK_LABELS = {
    "two_moons": "two_moons",
    "gaussian_mixture": "gaussian_mix",
    "gaussian_linear": "gaussian_lin",
    "gaussian_linear_uniform": "gaussian_lin_unif",
    "bernoulli_glm": "bernoulli_glm",
    "bernoulli_glm_raw": "bernoulli_glm_raw",
    "sir": "sir",
    "lotka_volterra": "lotka_volterra",
    "slcp": "slcp",
    "slcp_distractors": "slcp+distr",
    "two_moons_distractors": "two_moons+distr",
    "gaussian_mixture_distractors": "gaussian_mix+distr",
    "bernoulli_glm_distractors": "bernoulli_glm+distr",
    "sir_distractors": "sir+distr",
    "ar1_ts_t50": "ar1_t50",
    "ou": "ou",
    "solar_dynamo": "solar_dynamo",
    "sir_raw": "sir_raw",
    "lotka_volterra_raw": "lotka_volterra_raw",
}


def load_runs(base: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for path in sorted(base.glob("*_s*.npz")):
        data = dict(np.load(path, allow_pickle=True))
        task = str(data["task"])
        runs.setdefault(task, []).append(data)
    return runs


def summarize(runs: list[dict]) -> dict[str, np.ndarray]:
    ref_pin, r_mean, r_median = [], [], []
    for run in runs:
        best = int(run["best_layer"])
        ref_pin.append(float(run["pinball_ref"][best]))
        corr = np.asarray(run["corr"][best], dtype=float)
        taus = np.asarray(run["taus"], dtype=float)
        r_mean.append(float(np.nanmean(corr)))
        median_idx = int(np.argmin(np.abs(taus - 0.5)))
        r_median.append(float(corr[median_idx]))
    return {
        "ref_pin": np.asarray(ref_pin),
        "r_mean": np.asarray(r_mean),
        "r_median": np.asarray(r_median),
    }


def main() -> None:
    pfn = load_runs(PFN_DIR)
    raw = load_runs(RAW_DIR)
    tasks = [t for t in TASK_LABELS if t in pfn and t in raw]
    if not tasks:
        raise SystemExit("No overlapping PFN/raw quantile validation results.")

    pfn_s = {t: summarize(pfn[t]) for t in tasks}
    raw_s = {t: summarize(raw[t]) for t in tasks}

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 0.7 * len(tasks)), 4.6))
    xs = np.arange(len(tasks))
    w = 0.38

    for ax, metric, ylabel, title in [
        (axes[0], "r_mean", "Pearson r", "A. Mean quantile correlation"),
        (axes[1], "ref_pin", "Pinball loss", "B. Reference pinball"),
    ]:
        pfn_mean = np.array([pfn_s[t][metric].mean() for t in tasks])
        raw_mean = np.array([raw_s[t][metric].mean() for t in tasks])
        pfn_std = np.array([pfn_s[t][metric].std() for t in tasks])
        raw_std = np.array([raw_s[t][metric].std() for t in tasks])
        ax.bar(xs - w / 2, pfn_mean, w, yerr=pfn_std, label="PFN embedding", capsize=3)
        ax.bar(xs + w / 2, raw_mean, w, yerr=raw_std, label="raw x", capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if metric == "r_mean":
            ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False)

    fig.suptitle(f"Quantile-probe ablation: PFN embedding vs raw x ({len(tasks)} tasks)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "quantile_raw_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.pdf')}")

    print("\n=== Quantile raw comparison ===")
    print(f"{'task':<32} {'n_pfn':>5} {'n_raw':>5} {'pfn_r':>10} {'raw_r':>10} "
          f"{'pfn_pin':>10} {'raw_pin':>10}")
    for task in tasks:
        print(
            f"{task:<32} {len(pfn[task]):>5} {len(raw[task]):>5} "
            f"{pfn_s[task]['r_mean'].mean():>10.3f} "
            f"{raw_s[task]['r_mean'].mean():>10.3f} "
            f"{pfn_s[task]['ref_pin'].mean():>10.4f} "
            f"{raw_s[task]['ref_pin'].mean():>10.4f}"
        )


if __name__ == "__main__":
    main()
