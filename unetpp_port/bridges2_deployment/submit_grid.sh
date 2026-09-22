#!/bin/bash
# Submit only after the combined H100 gate has been reviewed and marked approved.
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
REPO=$ROOT/repo
GATE=$ROOT/calibration/H100_GATE_APPROVED.json

test -s "$GATE" || { echo "ERROR: missing reviewed H100 gate approval: $GATE" >&2; exit 1; }
"$ROOT/venv/bin/python" - <<PY
import json
p=json.load(open('$GATE'))
assert p['ok'] is True
assert sorted(p['passed_trainers']) == sorted([
  'unetpp_ds_on', 'unetpp_ds_off', 'plain_ds_on', 'plain_ds_off'
])
PY

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "ERROR: cluster checkout is dirty; refusing an ambiguous source snapshot" >&2; exit 1;
}
COMMIT=$(git -C "$REPO" rev-parse HEAD)
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)_${COMMIT:0:12}
echo "submitting commit=$COMMIT run=$RUN_ID"
mkdir -p "$ROOT/grid_logs"
sbatch --export="ALL,GRID_RETRY_COUNT=0,GRID_COMMIT=$COMMIT,GRID_RUN_ID=$RUN_ID" \
  "$REPO/unetpp_port/bridges2_deployment/grid_coordinated.sbatch"
