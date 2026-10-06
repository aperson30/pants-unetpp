#!/bin/bash
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=00:20:00
#SBATCH --job-name=pants_ct_dispatch_diag
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/ct_dispatch_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${DISPATCH_ROOT:?explicit pinned snapshot required}"
: "${DISPATCH_SHA256:?script hash required}"
SCRIPT=$DISPATCH_ROOT/evaluation/diagnose_ct_blas_dispatch.py
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$DISPATCH_SHA256"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
exec 9<"$RUN/.evaluation.lock"
flock -s -n 9 || exit 1
export PYTHONPATH=$DISPATCH_ROOT
timeout --kill-after=30s 1100s "$ROOT/venv/bin/python" -u "$SCRIPT" \
  --raw "$ROOT/ct_replay_diagnosis_47458101/PanTS_00009812/PanTS_00009812_0000.nii.gz" \
  --destination "$ROOT/ct_dispatch_diagnosis_${SLURM_JOB_ID}" \
  --frozen-correction "$ROOT/eval_frozen/5e99901263f4a9f2092d6674c7c566ba4d9d43cc/data_conversion/fix_affine_orthonormality.py" \
  --expected-sha256 02eca9599c5266c09e88d00238b73fe23b537633828d60e07211459b8bf3813f
