#!/bin/bash
# Prepared only: separate approval required for this tiny GPU reservation.
#SBATCH --account=cis260296p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time=00:03:00
#SBATCH --job-name=pants_original_cpu_replay
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/original_cpu_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${REPLAY_ROOT:?explicit pinned snapshot required}"
: "${REPLAY_SHA256:?script hash required}"
SCRIPT=$REPLAY_ROOT/evaluation/replay_ct_original_cpu.py
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$REPLAY_SHA256"
# Never force AVX512 onto an unsupported CPU. The scheduler's H100 nodes
# carry the original Sapphire Rapids CPU family; also check actual ISA flags.
grep -qm1 'avx512f' /proc/cpuinfo || exit 1
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
exec 9<"$RUN/.evaluation.lock"
flock -s -n 9 || exit 1
export PYTHONPATH=$REPLAY_ROOT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset OPENBLAS_CORETYPE
timeout --kill-after=5s 140s "$ROOT/venv/bin/python" -u "$SCRIPT" \
  --raw "$ROOT/ct_replay_diagnosis_47458101/PanTS_00009812/PanTS_00009812_0000.nii.gz" \
  --destination "$ROOT/original_cpu_replay_${SLURM_JOB_ID}" \
  --frozen-correction "$ROOT/eval_frozen/5e99901263f4a9f2092d6674c7c566ba4d9d43cc/data_conversion/fix_affine_orthonormality.py" \
  --raw-sha256 922ff5931c3344e0bf0cbc7134602f02e051381b6fd8c5dace1e20603d3a7cb3 \
  --correction-sha256 458a80d118e3458acd447a4d7bff8aaa7bd178a6bcb7125be3b2b991cbd3d729 \
  --evaluation-dir "$RUN"
