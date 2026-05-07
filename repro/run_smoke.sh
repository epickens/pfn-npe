#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-two_moons}"
SEED="${SEED:-42}"
N_TRAIN="${N_TRAIN:-128}"
N_VAL="${N_VAL:-64}"
EMBED_DIM="${EMBED_DIM:-16}"
DEVICE="${DEVICE:-cpu}"

uv run python -m pfn_testing.sbi.tabpfn_npe \
  --task "${TASK}" \
  --n-train "${N_TRAIN}" \
  --nval "${N_VAL}" \
  --seed "${SEED}" \
  --label-strategy per_dim \
  --model-type regressor \
  --embed-dim "${EMBED_DIM}" \
  --skip-raw \
  --device "${DEVICE}"
