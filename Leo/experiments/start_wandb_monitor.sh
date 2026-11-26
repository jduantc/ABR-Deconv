#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Wait up to 12 hours for a user-provided W&B login; never print credentials.
for ((attempt=0; attempt<2160; attempt++)); do
  if .venv/bin/python - <<'PY'
import netrc, os, sys
if os.environ.get('WANDB_API_KEY'):
    sys.exit(0)
try:
    credentials = netrc.netrc()
    sys.exit(0 if any(credentials.authenticators(host) for host in
                      ('api.forge.coreweave.com', 'api.wandb.ai', 'forge.coreweave.com')) else 1)
except (FileNotFoundError, netrc.NetrcParseError, OSError):
    sys.exit(1)
PY
  then
    exec .venv/bin/python experiments/wandb_monitor.py --output runs/comparison_16k
  fi
  sleep 20
done
printf 'W&B login was not found within 12 hours.\n' >&2
exit 1
