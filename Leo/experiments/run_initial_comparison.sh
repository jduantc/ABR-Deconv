#!/usr/bin/env bash
# The blanket study stops at 16k; longer training requires explicit selection.
set -euo pipefail
abr_leo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$abr_leo_dir"
abr_output=runs/comparison_16k
mkdir -p "$abr_output"
trap 'abr_exit=$?; if [[ $abr_exit -eq 0 ]]; then printf "COMPLETE\n"; else printf "FAILED (exit %s)\n" "$abr_exit"; fi > runs/comparison_16k/JOB_STATUS.txt' EXIT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
printf 'RUNNING — initial comparison capped at 16k updates\n' > "$abr_output/JOB_STATUS.txt"
.venv/bin/python experiments/check_gpu.py
.venv/bin/python -m pip freeze > "$abr_output/environment.txt"
sha256sum model.py data.py diffusion.py experiments/*.py > "$abr_output/source_sha256.txt"
.venv/bin/python experiments/study.py --device cuda --workers 1 \
    --output "$abr_output" --checkpoints 1000 4000 8000 12000 16000
