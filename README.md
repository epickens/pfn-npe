# PFN-NPE

**We now have a pre-print up!** It can be found here: https://arxiv.org/pdf/2605.07765

Research code for using frozen TabPFN representations as summary statistics for
simulation-based inference (SBI). The repository contains the PFN-NPE
implementation, comparison baselines, diagnostic experiments, and scripts used
to regenerate the main experiment tables and figures.

This is a research-code release, not a stable Python library. The command-line
entry points are intended to be run from a clone of this repository, and the
historical source namespace is `pfn_testing`.

## Repository Layout

- `pfn_testing/sbi/`: core task utilities, custom SBI tasks, PFN-NPE model code,
  normalizing-flow utilities, and baseline runners.
- `scripts/`: experiment, diagnostic, aggregation, and plotting entry points.
- `repro/`: shell wrappers for install checks, smoke tests, core sweeps, and
  table/figure regeneration.
- `results/`: default location for regenerated tables and lightweight summary
  artifacts.

Large generated arrays, trained weights, figures, logs, cluster job files, and
machine-specific paths are intentionally not tracked. New experiment outputs are
written under `pfn_testing/sbi/outputs/` and `results/`.

## Installation

The recommended workflow uses [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/epickens/pfn-npe.git
cd pfn-npe
uv sync
```

The dependency resolution is configured for Linux and macOS. The experiments
were developed for Linux GPU machines; Windows is not a supported target.
The environment uses a pinned `sbibm` compatibility fork because the public
`sbibm` release depends on an older `sbi` range than this code uses.

The NPE-PFN comparison wraps the upstream `mackelab/npe-pfn` implementation
through a pinned compatibility fork that supports the TabPFN and `sbi` versions
used here:

```bash
uv sync --extra npe-pfn
```

The full experiment suite is GPU-oriented. Small smoke runs can run on CPU, but
the 10k and 100k sweeps require many GPU-hours. TabPFN and SBIBM may download
model weights or benchmark assets on first use.

## Quick Checks

Verify that the environment imports and key command-line entry points are
available:

```bash
bash repro/check_install.sh
```

Run a tiny CPU PFN-NPE job:

```bash
DEVICE=cpu TASK=two_moons N_TRAIN=128 N_VAL=64 bash repro/run_smoke.sh
```

The smoke run writes outputs below `pfn_testing/sbi/outputs/`.

## Core Experiments

The fixed-budget comparison uses these task families:

- Standard SBIBM tasks:
  `two_moons gaussian_mixture gaussian_linear bernoulli_glm slcp sir lotka_volterra`
- Distractor variants:
  `two_moons_distractors gaussian_mixture_distractors bernoulli_glm_distractors sir_distractors`
- Time-series tasks:
  `ar1_ts_t50 ou solar_dynamo`

Run a single PFN-NPE cell:

```bash
uv run python -m pfn_testing.sbi.tabpfn_npe \
  --task two_moons \
  --n-train 10000 \
  --nval 2000 \
  --seed 42 \
  --label-strategy per_dim \
  --model-type regressor \
  --embed-dim 64 \
  --skip-raw \
  --device cuda
```

Run learned-summary NPE:

```bash
uv run python scripts/learned_summary_npe_baseline.py \
  --task two_moons --seed 42 --n-train 10000 --n-val 2000
```

Run NPE-PFN after installing the optional extra:

```bash
uv run python scripts/npe_pfn_baseline.py \
  --task two_moons --seed 42 --n-train 10000 --n-val 2000
```

Run standard `sbi` baselines:

```bash
uv run python -m pfn_testing.sbi.sbi_baselines \
  --task two_moons --method npe --n-train 10000 --seed 42
```

Run BayesFlow:

```bash
uv run python -m pfn_testing.sbi.bayesflow_baseline \
  --task two_moons --n-train 10000 --seed 42
```

The full 10k sweep wrapper is:

```bash
DEVICE=cuda RUN_NPE_PFN=1 bash repro/run_core_10k.sh
```

Restrict the sweep with environment variables:

```bash
TASKS="two_moons slcp" SEEDS="42" BUDGETS="10000" bash repro/run_core_10k.sh
```

## Diagnostics and Figures

PFN-NPE posterior-sample diagnostics require a trained PFN-NPE flow and a
quantile-probe reference cache:

```bash
uv run python scripts/validate_quantile_probe.py --task two_moons --seed 42
uv run python scripts/compare_flow_vs_quantile.py --task two_moons --seed 42
uv run python scripts/run_c2st_sweep.py --task two_moons --seed 42 --force
```

After the desired experiment cells have produced `flow_vs_quantile/*.npz`,
regenerate summary tables and figures:

```bash
bash repro/make_tables_and_figures.sh
```

## Artifact Map

| Component | Main code paths |
| --- | --- |
| Task simulation and references | `pfn_testing/sbi/sbibm_utils.py`, `pfn_testing/sbi/custom_tasks/` |
| PFN-NPE | `pfn_testing/sbi/tabpfn_npe.py`, `pfn_testing/sbi/density_estimators.py` |
| sbi and BayesFlow baselines | `pfn_testing/sbi/sbi_baselines.py`, `pfn_testing/sbi/bayesflow_baseline.py` |
| NPE-PFN baseline | `scripts/npe_pfn_baseline.py` |
| Learned-summary NPE | `scripts/learned_summary_npe_baseline.py` |
| Budget curves and C2ST tables | `scripts/run_c2st_sweep.py`, `scripts/plot_budget_scaling.py`, `scripts/paper_comparison_tables.py`, `scripts/aggregate_c2st_table.py` |
| Marginal, joint, and rank diagnostics | `scripts/c2st_decomposition.py`, `scripts/plot_c2st_marginal_joint_gap.py`, `scripts/plot_c2st_gap_compact.py` |
| Quantile and raw-observation probes | `scripts/validate_quantile_probe.py`, `scripts/validate_raw_quantile_probe.py`, `scripts/quantile_raw_ablation_table.py`, `scripts/plot_probe_quantile_two_panel.py` |
| Cross-theta probes | `scripts/cross_theta_probe.py`, `scripts/plot_cross_theta_layer_trajectory.py`, `scripts/plot_probe_main_and_role_table.py` |
| Filter, PCA, no-PCA XL, and TabPFN v2.5 ablations | `scripts/filtered_pfn_npe_flow.py`, `scripts/pca_no_pca_table.py`, `scripts/xl_no_pca_table.py`, `scripts/tabpfn_v25_comparison_table.py` |
| Wall-clock and anytime analyses | `scripts/wall_clock_amortization.py`, `scripts/wall_clock_table.py`, `scripts/run_anytime_runtime_sweep.py`, `scripts/plot_anytime_pareto_frontier.py` |
| Observation-specific local SBI | `scripts/run_targeted_local_sbi_benchmark.py`, `scripts/pfn_proposal_nle_refinement.py`, `scripts/pfn_weighted_fixed_table_nle.py` |

## Reproducibility Notes

Exact numeric reproduction depends on random seeds, package versions, GPU
backend, upstream model/data downloads, and successful completion of each
simulation and training cell. The code records generated artifacts locally but
does not bundle large outputs or pretrained experiment products.

For long runs, record the commit hash, `uv.lock` if present, GPU type, driver,
CUDA/PyTorch versions, and the exact command line. For public archival releases,
tag the commit used for each paper version.

## License

This repository is released under the MIT License. See `LICENSE`.
