#!/bin/bash
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=02:00:00
#SBATCH --job-name=pants_ct_replay_diag
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/ct_replay_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${DIAG_ROOT:?explicit pinned snapshot required}"
: "${DIAG_SHA256:?diagnostic script hash required}"
SCRIPT=$DIAG_ROOT/evaluation/diagnose_corrected_ct_replay.py
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$DIAG_SHA256"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
exec 9<"$RUN/.evaluation.lock"
flock -s -n 9 || exit 1
export PYTHONPATH=$ROOT/eval_frozen/5e99901263f4a9f2092d6674c7c566ba4d9d43cc:$DIAG_ROOT
# Match original evaluation preprocessing, not the audit's one-thread override.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout --kill-after=30s 7100s "$ROOT/venv/bin/python" -u "$SCRIPT" \
  --source-dir "$ROOT/source_geometry_recovery_20261005_47451639" \
  --evaluation-dir "$RUN" --destination "$ROOT/ct_replay_diagnosis_${SLURM_JOB_ID}"
