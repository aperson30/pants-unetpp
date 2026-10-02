#!/bin/bash
# Two locked additional patients, identical three arms. Diagnostic, not recall.
# Worst-case0.666667 charged GPUh. No automatic retry or outcome substitution.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --job-name=paired_judge_replicate
#SBATCH --output=/u/asanjeev/compression_probe_smalloutputs/paired_judge_replication_v1/replication_%j.log
set -euo pipefail
CODE=/u/asanjeev/compression_probe_smalloutputs/paired_judge_replication_v1
ROOT=/projects/bdyo/asanjeev/compression_probe_20261001
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export PYTHONPATH=$CODE/deps:$ROOT/wheels/monai-1.5.1-py3-none-any.whl:$CODE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 nnUNet_compile=False
cd "$CODE"
sha256sum -c source.sha256
failed=0
for study in 100430_00001 100259_00001; do
  echo CASE_START "$study"
  if srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
    timeout --signal=TERM --kill-after=10 480 "$PYTHON" -u paired_judge_pilot.py \
    --case "$CODE/case_$study" \
    --max-voxels 56000000 \
    --models /u/asanjeev/compression_probe_smalloutputs/judge_stage_fold4_v1/models \
    --vae "$ROOT/inputs/autoencoder_v1.pt" \
    --vae-sha 1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3 \
    --output "$CODE/results_${SLURM_JOB_ID}_$study"; then
    test -f "$CODE/results_${SLURM_JOB_ID}_$study/completion.json"
    echo CASE_COMPLETED "$study"
  else
    failed=1
    echo CASE_FAILED_NO_RETRY "$study"
  fi
done
if test "$failed" -eq 0; then echo REPLICATION_COMPLETE_TWO_CASES; fi
exit "$failed"
