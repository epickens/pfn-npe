"""Cross-task summary: trained flow vs linear-QR probe on marginal quantiles.

Loads `flow_vs_quantile/{task}_s{seed}.npz` outputs and produces a 3-panel
figure. The motivating question: is the flow wasting marginal information
that a 384-dim linear probe captures? If flow ≈ QR on marginals, the C2ST
gap is in the joint posterior structure, not the flow's marginal capacity.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CMP_DIR = Path("pfn_testing/sbi/outputs/layer_ablation/flow_vs_quantile")
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
    ("slcp", "slcp"),
    ("two_moons_distractors", "two_moons+distr"),
    ("gaussian_mixture_distractors", "gaussian_mixture+distr"),
    ("bernoulli_glm_distractors", "bernoulli_glm+distr"),
    ("sir_distractors", "sir+distr"),
]


def main() -> None:
    rows = []
    for task, label in TASK_ORDER:
        for p in sorted(CMP_DIR.glob(f"{task}_s*.npz")):
            d = dict(np.load(p, allow_pickle=True))
            rows.append({
                "task": task, "label": label, "seed": int(d["seed"]),
                "flow_pinball": float(d["flow_pinball"]),
                "qr_pinball": float(d["qr_pinball"]),
                "pearson_flow": d["pearson_flow"],          # (n_tau, dim_theta)
                "pearson_qr": d["pearson_qr"],
                "rmse_flow": d["rmse_flow"],
                "rmse_qr": d["rmse_qr"],
                "taus": d["taus"],
            })

    if not rows:
        print("No flow_vs_quantile results found.")
        return

    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    task_names = [t for t, _ in TASK_ORDER if t in by_task]
    task_labels = {t: lab for t, lab in TASK_ORDER}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── A: pinball comparison per task with seed error bars ──
    ax = axes[0]
    xs = np.arange(len(task_names))
    w = 0.38
    flow_means = [np.mean([r["flow_pinball"] for r in by_task[t]]) for t in task_names]
    flow_std = [np.std([r["flow_pinball"] for r in by_task[t]]) for t in task_names]
    qr_means = [np.mean([r["qr_pinball"] for r in by_task[t]]) for t in task_names]
    qr_std = [np.std([r["qr_pinball"] for r in by_task[t]]) for t in task_names]
    ax.bar(xs - w/2, flow_means, w, yerr=flow_std, color="C2",
           label="trained flow", capsize=3)
    ax.bar(xs + w/2, qr_means, w, yerr=qr_std, color="C1",
           label="linear-QR probe", capsize=3)
    for i, t in enumerate(task_names):
        for r in by_task[t]:
            ax.scatter(i - w/2, r["flow_pinball"], s=10, color="black",
                       alpha=0.6, zorder=3)
            ax.scatter(i + w/2, r["qr_pinball"], s=10, color="black",
                       alpha=0.6, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels([task_labels[t] for t in task_names],
                                          rotation=20, ha="right")
    ax.set_ylabel("Pinball loss of true θ on ref obs")
    ax.set_title(f"A. Marginal pinball: flow vs QR (n={len(rows)})")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # ── B: head-to-head Pearson r per task averaged across (τ, dim, seed) ──
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_names)))
    for color, t in zip(colors, task_names, strict=False):
        for r in by_task[t]:
            x_vals = np.nanmean(r["pearson_qr"])
            y_vals = np.nanmean(r["pearson_flow"])
            ax.scatter(x_vals, y_vals, color=color, s=80, edgecolor="black",
                       lw=0.5, alpha=0.85,
                       label=task_labels[t] if r is by_task[t][0] else None)
    ax.plot([-0.1, 1.05], [-0.1, 1.05], "--", color="grey", alpha=0.6, label="y=x")
    ax.set_xlabel("Linear-QR Pearson r (avg over τ, dim)")
    ax.set_ylabel("Flow-marginal Pearson r (avg over τ, dim)")
    ax.set_title("B. Head-to-head correlation against MCMC")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

    # ── C: per-τ correlation, mean across (task × seed × dim) ──
    ax = axes[2]
    taus_ref = rows[0]["taus"]
    flow_per_tau = []
    qr_per_tau = []
    for t_i in range(len(taus_ref)):
        flow_t = []
        qr_t = []
        for r in rows:
            flow_t.extend(r["pearson_flow"][t_i].tolist())
            qr_t.extend(r["pearson_qr"][t_i].tolist())
        flow_per_tau.append(flow_t)
        qr_per_tau.append(qr_t)
    flow_mean = [np.nanmean(x) for x in flow_per_tau]
    flow_std = [np.nanstd(x) for x in flow_per_tau]
    qr_mean = [np.nanmean(x) for x in qr_per_tau]
    qr_std = [np.nanstd(x) for x in qr_per_tau]
    ax.errorbar(taus_ref, flow_mean, yerr=flow_std, color="C2",
                marker="o", lw=2, label="trained flow", capsize=3)
    ax.errorbar(taus_ref, qr_mean, yerr=qr_std, color="C1",
                marker="s", lw=2, label="linear-QR probe", capsize=3)
    ax.set_xlabel("Quantile τ")
    ax.set_ylabel("Pearson r vs MCMC")
    ax.set_title("C. Per-τ correlation, mean across all (task, seed, dim)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    fig.suptitle(
        "Trained NSF flow vs linear-QR probe on marginal posterior quantiles",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "flow_vs_quantile_summary.png"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Wrote {out}")

    print("\n=== Per (task, seed) summary ===")
    print(f"{'task':<32} {'seed':>4} {'flow_pin':>9} {'qr_pin':>9} "
          f"{'flow_r':>9} {'qr_r':>9} {'Δr':>9}")
    for r in rows:
        flow_r = float(np.nanmean(r["pearson_flow"]))
        qr_r = float(np.nanmean(r["pearson_qr"]))
        print(f"{r['task']:<32} {r['seed']:>4} {r['flow_pinball']:>9.4f} "
              f"{r['qr_pinball']:>9.4f} {flow_r:>+9.3f} {qr_r:>+9.3f} "
              f"{flow_r - qr_r:>+9.3f}")


if __name__ == "__main__":
    main()
