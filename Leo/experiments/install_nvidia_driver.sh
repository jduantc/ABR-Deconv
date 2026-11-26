#!/usr/bin/env bash
# Run in a real terminal: sudo authenticates locally, never through chat.
set -euo pipefail
if [[ ${EUID} -ne 0 ]]; then
    exec sudo bash "$0" "$@"
fi
if [[ ! -f /etc/debian_version ]]; then
    echo 'This installer is intended for this Debian machine.' >&2
    exit 1
fi
abr_backup="/etc/apt/sources.list.abr-backup-$(date +%Y%m%d-%H%M%S)"
cp -a /etc/apt/sources.list "$abr_backup"
python3 - <<'PY'
from pathlib import Path
p=Path('/etc/apt/sources.list')
lines=[]
for line in p.read_text().splitlines():
    if line.startswith(('deb ', 'deb-src ')) and any(
        host in line for host in ('deb.debian.org/', 'security.debian.org/')):
        text, sep, comment=line.partition('#')
        fields=text.split()
        for component in ('contrib','non-free','non-free-firmware'):
            if component not in fields:
                fields.append(component)
        line=' '.join(fields)+(' #'+comment if sep else '')
    lines.append(line)
p.write_text('\n'.join(lines)+'\n')
PY
echo "APT source backup: $abr_backup"
apt-get update
# Explicitly request the proprietary DKMS module, not the open 550 module
# that Debian documents as incompatible with recent 6.12 kernels.
apt-get install -y "linux-headers-$(uname -r)" nvidia-kernel-dkms nvidia-driver firmware-misc-nonfree
dkms status
if ! modinfo -k "$(uname -r)" nvidia >/dev/null 2>&1 &&
   ! modinfo -k "$(uname -r)" nvidia-current >/dev/null 2>&1; then
    echo 'NVIDIA module is not built for the current kernel. Resolve the DKMS errors before rebooting.' >&2
    exit 1
fi
echo 'Driver installation completed. Save your work, reboot from the desktop, then run nvidia-smi.'
echo 'This script does not reboot or unload your current display driver.'
