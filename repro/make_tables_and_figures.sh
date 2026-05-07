#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/tables

uv run python scripts/aggregate_c2st_table.py

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
