#!/usr/bin/env bash
set -euo pipefail
abr_leo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$abr_leo_dir"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Official CUDA 12.4 build compatible with the Debian Trixie 550 driver family.
.venv/bin/python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r experiments/requirements.txt
.venv/bin/python experiments/check_gpu.py
