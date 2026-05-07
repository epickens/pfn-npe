"""Cross-task comparison: raw TabPFN encoder vs contrastive (v3) encoder.

For each task at seed 42, loads default-NSF flow samples for both encoder
variants and the corresponding c2st_decomp output. Plots 4 panels (joint,
marginal, rank C2ST + pinball) with side-by-side bars per task.

If the contrastive encoder gives consistent improvements on the tasks it
was trained on (likely distractor variants), the bar pattern surfaces it.
If it regresses on slcp (out-of-distribution), that's also visible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DECOMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/c2st_decomp")
SAMPLE_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
FIG_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TASK_ORDER = [
    ("two_moons_distractors", "two_moons+distr"),
    ("gaussian_mixture_distractors", "gaussian_mixture+distr"),
    ("bernoulli_glm_distractors", "bernoulli_glm+distr"),
    ("sir_distractors", "sir+distr"),
    ("slcp_distractors", "slcp+distr"),
    ("slcp", "slcp"),
]


def load_metric(task: str, seed: int, suffix: str) -> dict | None:
    decomp_p = DECOMP_DIR / f"{task}_s{seed}{suffix}.npz"
    sample_p = SAMPLE_DIR / f"{task}_s{seed}{suffix}.npz"
    if not (decomp_p.exists() and sample_p.exists()):
        return None
    decomp = dict(np.load(decomp_p, allow_pickle=True))
    sample = dict(np.load(sample_p, allow_pickle=True))
    return {
        "joint": float(decomp["joint"].mean()),
        "marg": float(decomp["marginal"].mean()),
        "rank": float(decomp["rank"].mean()),
        "pinball": float(sample.get("flow_pinball", float("nan"))),
        "joint_std": float(decomp["joint"].std()),
        "rank_std": float(decomp["rank"].std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enc-tag", default="v3_step190k")
    args = ap.parse_args()

    rows = []
    for task, label in TASK_ORDER:
        # Raw TabPFN: prefer fresh-NSF retrain (slcp_distractors only) else saved NSF (default).
        raw_suffix_candidates = ["_nsf", ""]
        raw = None
        for suffix in raw_suffix_candidates:
            raw = load_metric(task, args.seed, suffix)
            if raw is not None:
                break
        v3 = load_metric(task, args.seed, f"_nsf_enc_{args.enc_tag}")
        if raw is None or v3 is None:
            print(f"  [skip] {task}: raw={raw is not None}, v3={v3 is not None}")
            continue
        rows.append((task, label, raw, v3))

    if not rows:
        print("No comparison data found.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    metrics = [("joint", "Joint C2ST"), ("marg", "Marginal C2ST"),
               ("rank", "Rank C2ST (copula)"), ("pinball", "Pinball")]
    xs = np.arange(len(rows))
    w = 0.38

    for (key, title), ax in zip(metrics, axes.flatten(), strict=True):
        raw_vals = [r[key] for _, _, r, _ in rows]
        v3_vals = [v[key] for _, _, _, v in rows]
        ax.bar(xs - w/2, raw_vals, w, color="C0", label="raw TabPFN")
        ax.bar(xs + w/2, v3_vals, w, color="C3", label=f"contrastive ({args.enc_tag})")
        if key in ("joint", "marg", "rank"):
            ax.axhline(0.5, color="grey", ls=":", alpha=0.7,
                       label="C2ST = 0.5 (ideal)")
        for i, (rv, vv) in enumerate(zip(raw_vals, v3_vals, strict=True)):
            delta = vv - rv
            color = "green" if delta < 0 else "red"
            ax.text(i, max(rv, vv) + 0.005, f"{delta:+.3f}", ha="center",
                    fontsize=7, color=color, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab, _, _ in rows], rotation=20,
                           ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Encoder comparison: raw TabPFN vs contrastive ({args.enc_tag}), "
        f"default NSF | seed={args.seed}",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / f"encoder_comparison_{args.enc_tag}_s{args.seed}.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")

    print("\n=== Per-task encoder comparison ===")
    print(f"{'task':<32} {'metric':>10} {'raw':>9} {'v3':>9} {'Δ':>9}")
    for task, _, raw, v3 in rows:
        for k, _ in metrics:
            d = v3[k] - raw[k]
            print(f"{task:<32} {k:>10} {raw[k]:>9.4f} {v3[k]:>9.4f} {d:>+9.4f}")


if __name__ == "__main__":
    main()
