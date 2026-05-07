#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-two_moons gaussian_mixture gaussian_linear bernoulli_glm slcp sir lotka_volterra two_moons_distractors gaussian_mixture_distractors bernoulli_glm_distractors sir_distractors ar1_ts_t50 ou solar_dynamo}"
SEEDS="${SEEDS:-42 123 7}"
BUDGETS="${BUDGETS:-10000}"
DEVICE="${DEVICE:-cuda}"
N_VAL="${N_VAL:-2000}"
N_REF="${N_REF:-10}"
N_FLOW_SAMPLES="${N_FLOW_SAMPLES:-1000}"

RUN_PFN_NPE="${RUN_PFN_NPE:-1}"
RUN_LEARNED="${RUN_LEARNED:-1}"
RUN_NPE_PFN="${RUN_NPE_PFN:-0}"
RUN_C2ST="${RUN_C2ST:-1}"

for budget in ${BUDGETS}; do
  if [ "${budget}" != "10000" ]; then
    echo "run_core_10k.sh is intended for the 10k manuscript table."
    echo "For other budgets, write separate output directories or add n-suffixed artifact handling."
    exit 2
  fi

  for task in ${TASKS}; do
    for seed in ${SEEDS}; do
      echo "=== task=${task} seed=${seed} n_train=${budget} ==="

      if [ "${RUN_PFN_NPE}" = "1" ]; then
        uv run python -m pfn_testing.sbi.tabpfn_npe \
          --task "${task}" \
          --n-train "${budget}" \
          --nval "${N_VAL}" \
          --seed "${seed}" \
          --label-strategy per_dim \
          --model-type regressor \
          --embed-dim 64 \
          --skip-raw \
          --device "${DEVICE}"

        uv run python scripts/validate_quantile_probe.py \
          --task "${task}" \
          --seed "${seed}" \
          --n-train "${budget}" \
          --n-val "${N_VAL}" \
          --n-ref "${N_REF}"

        uv run python scripts/compare_flow_vs_quantile.py \
          --task "${task}" \
          --seed "${seed}" \
          --n-train "${budget}" \
          --n-val "${N_VAL}" \
          --n-flow-samples "${N_FLOW_SAMPLES}" \
          --n-ref "${N_REF}"
      fi

      if [ "${RUN_LEARNED}" = "1" ]; then
        uv run python scripts/learned_summary_npe_baseline.py \
          --task "${task}" \
          --seed "${seed}" \
          --n-train "${budget}" \
          --n-val "${N_VAL}" \
          --n-flow-samples "${N_FLOW_SAMPLES}" \
          --n-ref "${N_REF}"
      fi

      if [ "${RUN_NPE_PFN}" = "1" ]; then
        uv run python scripts/npe_pfn_baseline.py \
          --task "${task}" \
          --seed "${seed}" \
          --n-train "${budget}" \
          --n-val "${N_VAL}" \
          --n-flow-samples "${N_FLOW_SAMPLES}" \
          --n-ref "${N_REF}"
      fi

      if [ "${RUN_C2ST}" = "1" ]; then
        uv run python scripts/run_c2st_sweep.py \
          --task "${task}" \
          --seed "${seed}" \
          --n-ref "${N_REF}" \
          --force
      fi
    done
  done
done
