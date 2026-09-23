#!/bin/bash
# Intentionally inert until a reviewed DeltaAI gate file is created.
set -euo pipefail
PROJECT=/projects/bdyo/asanjeev
REPO=$PROJECT/pants-unetpp-fork
ROOT=/work/nvme/bdyo/asanjeev/pants_grid_deltaai
GATE=$ROOT/DELTAI_GATE_APPROVED.json
test -s "$GATE" || { echo "missing reviewed DeltaAI launch gate: $GATE" >&2; exit 1; }
COMMIT=$(git -C "$REPO" rev-parse HEAD)
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet
"$PROJECT/pants_venv/bin/python" - "$GATE" "$COMMIT" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1]))
assert p['ok'] is True
assert p['commit'] == sys.argv[2]
assert p['stack'] == 'torch-2.10.0+cu129-nnunetv2-2.8.1'
assert p['passed_trainers'] == [
    'nnUNetTrainerUNetPlusPlusSparseValidation',
    'nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation',
    'nnUNetTrainerBF16SparseValidation',
    'nnUNetTrainerBF16NoDeepSupervisionSparseValidation',
]
PY
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)_${COMMIT:0:12}
mkdir -p "$ROOT/logs"
echo "ready to submit commit=$COMMIT run=$RUN_ID"
sbatch --export="ALL,GRID_RETRY_COUNT=0,GRID_COMMIT=$COMMIT,GRID_RUN_ID=$RUN_ID" \
  "$REPO/unetpp_port/deltaai_deployment/grid_coordinated.sbatch"
