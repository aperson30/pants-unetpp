#!/bin/bash
# CPU-only final scoring. Submit afterok of completed four-cell inference.
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=10:00:00
#SBATCH --job-name=pants_versioned_grid_score
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/versioned_score_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${SCORING_ROOT:?pinned snapshot required}"
: "${SCORING_SHA256:?scorer hash required}"
: "${SOURCE_AUDIT_JOB:?successful source audit required}"
: "${REFERENCE_JOB:?successful reference builder required}"
SCRIPT=$SCORING_ROOT/evaluation/score_versioned_grid_cpu.py
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$SCORING_SHA256"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
SOURCE=$ROOT/source_901_audit_${SOURCE_AUDIT_JOB}/all_901_source_audit.json
REFERENCE=$ROOT/tumor_reference_v1_${REFERENCE_JOB}
METRIC=$ROOT/eval_frozen/5e99901263f4a9f2092d6674c7c566ba4d9d43cc/evaluation/compute_tumor_metrics.py
test -f "$SOURCE"
test -f "$REFERENCE/tumor_reference_manifest.json"
test -f "$METRIC"
export PYTHONPATH=$SCORING_ROOT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# CLI holds the existing shared evaluation lock and performs full pre/post audits.
# All four complete cohorts required. A timeout/failure never publishes the table.
timeout --kill-after=30s 35900s "$ROOT/venv/bin/python" -u -m evaluation.score_versioned_grid_cpu \
  --evaluation-dir "$RUN" --reference-dir "$REFERENCE" --source-audit "$SOURCE" \
  --metric-script "$METRIC" --destination "$ROOT/versioned_grid_scores_${SLURM_JOB_ID}"
