#!/bin/bash
# Only the locked smaller additional patient. Larger case remains blocked.
# No cropping, no substitution, no automatic retry. Max0.333333 charged GPUh.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --job-name=paired_judge_small
#SBATCH --output=/u/asanjeev/compression_probe_smalloutputs/paired_judge_replication_small_v1/replication_%j.log
set -euo pipefail
CODE=/u/asanjeev/compression_probe_smalloutputs/paired_judge_replication_small_v1
ROOT=/projects/bdyo/asanjeev/compression_probe_20261001
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export PYTHONPATH=$CODE/deps:$ROOT/wheels/monai-1.5.1-py3-none-any.whl:$CODE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 nnUNet_compile=False
cd "$CODE"
sha256sum -c source.sha256
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=10 540 "$PYTHON" -u paired_judge_pilot.py \
 --case "$CODE/case_100430_00001" --max-voxels 33000000 \
 --models /u/asanjeev/compression_probe_smalloutputs/judge_stage_fold4_v1/models \
 --vae "$ROOT/inputs/autoencoder_v1.pt" \
 --vae-sha 1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3 \
 --output "$CODE/results_${SLURM_JOB_ID}_100430_00001"
test -f "$CODE/results_${SLURM_JOB_ID}_100430_00001/completion.json"
echo SMALL_REPLICATION_COMPLETE_LARGER_CASE_STILL_BLOCKED
