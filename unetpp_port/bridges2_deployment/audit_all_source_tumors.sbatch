#!/bin/bash
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=02:00:00
#SBATCH --job-name=pants_901_source_audit
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/source_901_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${AUDIT_ROOT:?explicit pinned snapshot required}"
: "${AUDIT_SHA256:?script hash required}"
SCRIPT=$AUDIT_ROOT/evaluation/audit_all_source_tumors.py
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$AUDIT_SHA256"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
exec 9<"$RUN/.evaluation.lock"
flock -s -n 9 || exit 1
export PYTHONPATH=$ROOT/eval_frozen/5e99901263f4a9f2092d6674c7c566ba4d9d43cc:$AUDIT_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
EXTRA_ARGS=()
if [[ -n ${EXACT_CT_REPLAY_DIR:-} ]]; then
  test -d "$EXACT_CT_REPLAY_DIR"
  EXTRA_ARGS+=(--exact-ct-replay-dir "$EXACT_CT_REPLAY_DIR")
fi
timeout --kill-after=30s 7100s "$ROOT/venv/bin/python" -u "$SCRIPT" \
  --source-dir "$ROOT/source_geometry_recovery_20261005_47451639" \
  --evaluation-dir "$RUN" --report-dir "$ROOT/source_901_audit_${SLURM_JOB_ID}" "${EXTRA_ARGS[@]}"
