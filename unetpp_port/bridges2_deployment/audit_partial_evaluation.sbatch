#!/bin/bash
# Integrity-only audit of saved evaluation work: no GPUs or test metrics.
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=02:00:00
#SBATCH --job-name=pants_eval_resume_audit
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/resume_audit_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${AUDIT_COMMIT:?supply the frozen audit revision}"
[[ "$AUDIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 1
AUDIT=$ROOT/eval_audit_frozen/$AUDIT_COMMIT
test "$(cat "$AUDIT/.frozen_commit")" = "$AUDIT_COMMIT"
TRAIN=61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
ORIGINAL=5e99901263f4a9f2092d6674c7c566ba4d9d43cc
RUN=$ROOT/results/evaluation_grid/$TRAIN
# Shared lock: refuse an audit while a writer is active. Never alter this lock file.
exec 9<"$RUN/.evaluation.lock"
flock -s -n 9 || { echo 'evaluation writer active'; exit 1; }
export PYTHONPATH=$ROOT/eval_frozen/$ORIGINAL
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$AUDIT"
date -u
"$ROOT/venv/bin/python" -u "$AUDIT/evaluation/audit_partial_resume.py" \
  --evaluation-dir "$RUN" --frozen-evaluation "$ROOT/eval_frozen/$ORIGINAL"
echo PARTIAL_RESUME_INTEGRITY_PASSED
