#!/bin/bash
# CPU-only read-only resume gate. No inference, scoring or artifact retirement.
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=01:00:00
#SBATCH --job-name=pants_partial_resume_audit
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/partial_resume_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${AUDIT_ROOT:?pinned snapshot required}"
: "${AUDIT_SHA256:?auditor hash required}"
: "${SOURCE_AUDIT_JOB:?successful source audit required}"
: "${REFERENCE_JOB:?successful reference builder required}"
SCRIPT=$AUDIT_ROOT/evaluation/audit_versioned_predictions.py
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$AUDIT_SHA256"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
SOURCE=$ROOT/source_901_audit_${SOURCE_AUDIT_JOB}/all_901_source_audit.json
REFERENCE=$ROOT/tumor_reference_v1_${REFERENCE_JOB}
test -f "$SOURCE"
test -f "$REFERENCE/tumor_reference_manifest.json"
export PYTHONPATH=$AUDIT_ROOT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout --kill-after=30s 3500s "$ROOT/venv/bin/python" -u -m evaluation.audit_versioned_predictions \
  --evaluation-dir "$RUN" --reference-dir "$REFERENCE" --source-audit "$SOURCE" \
  --partial --report "$ROOT/partial_resume_audit_${SLURM_JOB_ID}.json"
