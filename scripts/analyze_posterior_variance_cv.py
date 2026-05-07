"""Quantify how much posterior variance varies across sbibm reference observations.

For each task, load the reference posterior samples at each of the 10 sbibm
reference observations, compute per-dim Var, then compute coefficient of
variation across observations. The CV predicts how much heteroscedastic
signal the variance probe should be able to find — tasks with near-constant
posterior variance have little signal to learn regardless of the encoder.

Writes a CSV + a scatter plot of ∆NLL (from the variance probe) vs posterior-
variance CV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.sbibm_utils import get_task  # noqa: E402

TASKS_WITH_DISTRACTOR_PROBE = [
    # (task_for_probe_npz, base_task_for_reference_posterior)
    ("two_moons_distractors", "two_moons"),
    ("sir_distractors", "sir"),
    ("slcp", "slcp"),
    ("slcp_distractors", "slcp"),
    ("bernoulli_glm_distractors", "bernoulli_glm"),
    ("bernoulli_glm_raw", "bernoulli_glm"),
    ("gaussian_linear_uniform", "gaussian_linear_uniform"),
    ("gaussian_mixture_distractors", "gaussian_mixture"),
]

PROBE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/probe")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")
SEED = 42


def posterior_variance_cv(base_task_name: str, n_ref: int = 10) -> tuple[float, np.ndarray]:
    """Return (mean CV across dims, per-ref-obs per-dim variance matrix)."""
    task = get_task(base_task_name)
    per_dim_var = []
    for i in range(1, n_ref + 1):
        try:
            samples = task.get_reference_posterior_samples(num_observation=i).numpy()
        except Exception:
            break
        per_dim_var.append(samples.var(axis=0))
    if not per_dim_var:
        return float("nan"), np.zeros((0, 0))
    v = np.stack(per_dim_var)  # (n_ref, dim_theta)
    cv = v.std(axis=0) / (v.mean(axis=0) + 1e-8)
    return float(cv.mean()), v


def load_best_delta_nll(probe_task: str) -> float | None:
    p = PROBE_DIR / f"{probe_task}_s{SEED}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "nll" not in d:
        return None
    gap = d["nll"] - d["nll_homo"]
    return float(gap.min())


def main() -> None:
    rows = []
    for probe_task, base_task in TASKS_WITH_DISTRACTOR_PROBE:
        cv, _ = posterior_variance_cv(base_task)
        dnll = load_best_delta_nll(probe_task)
        rows.append((probe_task, base_task, cv, dnll))
        print(f"{probe_task:<35} base={base_task:<25} CV={cv:.3f}  ∆NLL={dnll}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for probe_task, base_task, cv, dnll in rows:
        if dnll is None or np.isnan(cv):
            continue
        ax.scatter(cv, dnll, s=60, alpha=0.85)
        ax.annotate(probe_task.replace("_distractors", "*"),
                    (cv, dnll), fontsize=7,
                    xytext=(6, 2), textcoords="offset points")
    ax.axhline(0, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("Coefficient of variation of posterior Var across ref obs")
    ax.set_ylabel(r"$\Delta$ NLL at best layer (negative = probe wins)")
    ax.set_title("Heteroscedastic probe gain vs task-intrinsic variance variation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "posterior_variance_cv_vs_delta_nll.png"
    fig.savefig(str(out), dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
