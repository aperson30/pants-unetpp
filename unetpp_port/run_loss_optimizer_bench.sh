#!/usr/bin/env bash
set -euo pipefail

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU_BUSY: refusing to launch" >&2
    exit 20
fi

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 70 * 1024 * 1024 )); then
    echo "LOW_MEMORY: ${available_kib} KiB available; refusing to launch" >&2
    exit 21
fi

cd "$(dirname "$0")"
exec timeout 900 "$HOME/.conda/envs/pants_unetpp/bin/python" loss_optimizer_bench.py "$@"
