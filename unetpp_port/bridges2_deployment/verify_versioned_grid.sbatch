#!/bin/bash
# Read-only CPU check, afterok of the versioned CPU scorer.
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=02:00:00
#SBATCH --job-name=pants_final_report_verify
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/final_report_verify_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${VERIFY_ROOT:?pinned snapshot required}"
: "${VERIFY_SHA256:?verifier hash required}"
: "${SOURCE_AUDIT_JOB:?successful source audit required}"
: "${REFERENCE_JOB:?successful reference builder required}"
: "${SCORING_JOB:?successful scorer required}"
SCRIPT=$VERIFY_ROOT/evaluation/verify_versioned_grid_report.py
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$VERIFY_SHA256"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
SOURCE=$ROOT/source_901_audit_${SOURCE_AUDIT_JOB}/all_901_source_audit.json
REFERENCE=$ROOT/tumor_reference_v1_${REFERENCE_JOB}
SCORES=$ROOT/versioned_grid_scores_${SCORING_JOB}
METRIC=$ROOT/eval_frozen/5e99901263f4a9f2092d6674c7c566ba4d9d43cc/evaluation/compute_tumor_metrics.py
test -f "$SCORES/grid_metrics.json"
export PYTHONPATH=$VERIFY_ROOT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout --kill-after=30s 7100s "$ROOT/venv/bin/python" -u -m evaluation.verify_versioned_grid_report \
  --evaluation-dir "$RUN" --reference-dir "$REFERENCE" --source-audit "$SOURCE" \
  --metric-script "$METRIC" --scores-dir "$SCORES"
