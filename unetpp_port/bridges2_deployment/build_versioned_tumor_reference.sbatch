#!/bin/bash
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=01:00:00
#SBATCH --job-name=pants_tumor_reference
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/tumor_reference_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${REFERENCE_ROOT:?explicit pinned snapshot required}"
: "${REFERENCE_SHA256:?builder hash required}"
: "${SOURCE_AUDIT_JOB:?completed full-cohort CPU job required}"
: "${SOURCE_AUDIT_SCRIPT_SHA256:?source audit script hash required}"
SCRIPT=$REFERENCE_ROOT/evaluation/build_versioned_tumor_reference.py
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$REFERENCE_SHA256"
AUDIT=$ROOT/source_901_audit_${SOURCE_AUDIT_JOB}/all_901_source_audit.json
test -f "$AUDIT"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
exec 9<"$RUN/.evaluation.lock"
flock -s -n 9 || exit 1
export PYTHONPATH=$REFERENCE_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ROOT/venv/bin/python" -c 'import json, os, sys; report=json.load(open(sys.argv[1])); assert report["script_sha256"] == os.environ["SOURCE_AUDIT_SCRIPT_SHA256"]' "$AUDIT"
AUDIT_HASH=$(sha256sum "$AUDIT" | cut -d ' ' -f1)
timeout --kill-after=30s 3500s "$ROOT/venv/bin/python" -u "$SCRIPT" \
  --evaluation-dir "$RUN" --source-audit "$AUDIT" \
  --source-audit-sha256 "$AUDIT_HASH" \
  --destination "$ROOT/tumor_reference_v1_${SLURM_JOB_ID}"
