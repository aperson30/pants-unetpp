#!/bin/bash
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=00:15:00
#SBATCH --job-name=pants_ct_header_diag
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/ct_header_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${HEADER_ROOT:?explicit pinned snapshot required}"
: "${HEADER_SHA256:?script hash required}"
SCRIPT=$HEADER_ROOT/evaluation/diagnose_ct_header_replay.py
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$HEADER_SHA256"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
exec 9<"$RUN/.evaluation.lock"
flock -s -n 9 || exit 1
export PYTHONPATH=$HEADER_ROOT OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout --kill-after=30s 800s "$ROOT/venv/bin/python" -u "$SCRIPT" \
  --diagnosis-dir "$ROOT/ct_replay_diagnosis_47458101" \
  --evaluation-dir "$RUN" --destination "$ROOT/ct_header_diagnosis_${SLURM_JOB_ID}"
