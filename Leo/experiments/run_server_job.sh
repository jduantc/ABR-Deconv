#!/usr/bin/env bash
# Persistent, fully logged server run. Does not change system packages/drivers.
set -euo pipefail
abr_leo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$abr_leo_dir"
mkdir -p runs/comparison_gpu
printf 'VALIDATING\n' > runs/comparison_gpu/JOB_STATUS.txt
trap 'abr_exit=$?; if [[ $abr_exit -eq 0 ]]; then printf "COMPLETE\n"; else printf "FAILED (exit %s)\n" "$abr_exit"; fi > runs/comparison_gpu/JOB_STATUS.txt' EXIT
if [[ -n "${ABR_SETUP_PID:-}" ]]; then
    printf 'WAITING FOR PYTHON ENVIRONMENT INSTALLATION\n' > runs/comparison_gpu/JOB_STATUS.txt
    while kill -0 "$ABR_SETUP_PID" 2>/dev/null; do
        abr_setup_state="$(ps -p "$ABR_SETUP_PID" -o stat=)"
        if [[ "$abr_setup_state" == Z* ]]; then break; fi
        sleep 5
    done
fi
printf 'VALIDATING\n' > runs/comparison_gpu/JOB_STATUS.txt
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
date -Is
hostname
.venv/bin/python experiments/check_gpu.py
.venv/bin/python -m pytest experiments/test_study.py -q
sha256sum model.py data.py diffusion.py experiments/*.py > runs/comparison_gpu/source_sha256.txt
printf 'RUNNING — training, validation guidance sweep, held-out evaluation, plots\n' > runs/comparison_gpu/JOB_STATUS.txt
bash experiments/run_gpu.sh
date -Is
