#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/tables

uv run python scripts/aggregate_c2st_table.py
uv run python scripts/aggregate_c2st_table.py \
  --metric joint \
  --methods nsf learned_summary_npe npe_pfn \
  --tex-out results/tables/c2st_joint.tex

uv run python scripts/paper_comparison_tables.py \
  --external-tex-out results/tables/external_baselines_10k.tex \
  --compact-tex-out results/tables/compact_family_summary.tex

uv run python scripts/plot_budget_scaling.py --make-paper-variants
uv run python scripts/plot_c2st_marginal_joint_gap.py
uv run python scripts/plot_c2st_gap_compact.py
uv run python scripts/plot_probe_quantile_two_panel.py

uv run python scripts/quantile_raw_ablation_table.py
uv run python scripts/tabpfn_v25_comparison_table.py \
  --tex-dir results/tables

uv run python scripts/pca_no_pca_table.py \
  --tex-out results/tables/pca_no_pca_budget.tex

uv run python scripts/xl_no_pca_table.py \
  --csv-out results/tables/xl_no_pca_table.csv \
  --tex-out results/tables/xl_no_pca_table.tex

if compgen -G "pfn_testing/sbi/outputs/layer_ablation/c2st_decomp/*_n100000.npz" > /dev/null; then
  uv run python scripts/aggregate_c2st_table.py \
    --budget 100000 \
    --metric joint \
    --methods nsf learned_summary_npe npe_pfn \
    --tex-out results/tables/c2st_joint_100k.tex

  uv run python scripts/paper_comparison_tables.py \
    --budget 100000 \
    --external-tex-out results/tables/external_baselines_100k.tex \
    --compact-tex-out results/tables/compact_family_summary_100k.tex
fi

if compgen -G "pfn_testing/sbi/outputs/layer_ablation/c2st_decomp/*_nsf_filter_n100000.npz" > /dev/null; then
  uv run python scripts/p17_filter_ablation_table.py \
    --tex-out results/tables/p17_filter_ablation_joint.tex
fi

if compgen -G "pfn_testing/sbi/outputs/layer_ablation/cross_theta/*.npz" > /dev/null; then
  uv run python scripts/plot_cross_theta_layer_trajectory.py
  uv run python scripts/plot_probe_main_and_role_table.py
  uv run python scripts/plot_slcp_probe_case_study.py
fi

if [ -f "pfn_testing/sbi/outputs/layer_ablation/wall_clock/wall_clock.csv" ]; then
  uv run python scripts/wall_clock_table.py \
    --from-csv pfn_testing/sbi/outputs/layer_ablation/wall_clock/wall_clock.csv \
    --tex-out results/tables/wall_clock.tex
fi

if [ -f "pfn_testing/sbi/outputs/layer_ablation/wall_clock/wall_clock_amortization.csv" ]; then
  uv run python scripts/plot_wall_clock_amortization_runtime.py \
    --timing-csv pfn_testing/sbi/outputs/layer_ablation/wall_clock/wall_clock_amortization.csv
fi

if [ -f "pfn_testing/sbi/outputs/anytime_runtime/anytime_pareto_frontier_all_tasks_min3.csv" ] && \
   [ -f "pfn_testing/sbi/outputs/anytime_runtime/anytime_seed_points_all_tasks_min3.csv" ]; then
  uv run python scripts/plot_anytime_pareto_appendix_combined.py
fi
