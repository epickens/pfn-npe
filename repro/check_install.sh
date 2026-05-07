#!/usr/bin/env bash
set -euo pipefail

uv run python - <<'PY'
from pfn_testing.sbi.sbibm_utils import AVAILABLE_TASKS, get_task_info

print(f"available_tasks={len(AVAILABLE_TASKS)}")
print(get_task_info("two_moons"))
PY

uv run python -m pfn_testing.sbi.tabpfn_npe --help >/dev/null
uv run python -m pfn_testing.sbi.sbi_baselines --help >/dev/null
uv run python -m pfn_testing.sbi.bayesflow_baseline --help >/dev/null
uv run python scripts/run_c2st_sweep.py --help >/dev/null
uv run python scripts/validate_quantile_probe.py --help >/dev/null

echo "Install check passed."
