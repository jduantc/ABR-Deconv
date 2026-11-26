#!/usr/bin/env bash
set -euo pipefail
abr_leo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$abr_leo_dir"
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run bash experiments/setup_gpu_env.sh first.' >&2
    exit 1
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
.venv/bin/python experiments/check_gpu.py
# Reuse only source downloads; train all models from scratch for the GPU study.
.venv/bin/python - <<'PY'
from pathlib import Path
source=Path('runs/cpu_pilot/data/earndb_cache').resolve()
target=Path('runs/comparison_gpu/data/earndb_cache')
if source.exists() and not target.exists():
    target.parent.mkdir(parents=True,exist_ok=True)
    target.symlink_to(source,target_is_directory=True)
PY
mkdir -p runs/comparison_gpu
.venv/bin/python -m pip freeze > runs/comparison_gpu/environment.txt
.venv/bin/python experiments/study.py --device cuda --workers 1 \
    --output runs/comparison_gpu "$@" 2>&1 | tee -a runs/comparison_gpu/console.log
