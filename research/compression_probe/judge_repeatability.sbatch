#!/bin/bash
# Four same-input detector calls; no new scientific arm or automatic retry.
# Maximum0.1charge-equivalentGPUh at billing2000.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:03:00
#SBATCH --no-requeue
#SBATCH --job-name=judge_repeatability
#SBATCH --output=/u/asanjeev/compression_probe_smalloutputs/judge_repeatability_v1/repeat_%j.log
set -euo pipefail
CODE=/u/asanjeev/compression_probe_smalloutputs/judge_repeatability_v1
BASE=/u/asanjeev/compression_probe_smalloutputs/paired_judge_replication_small_v2
ROOT=/projects/bdyo/asanjeev/compression_probe_20261001
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export PYTHONPATH=$BASE/deps:$ROOT/wheels/monai-1.5.1-py3-none-any.whl:$CODE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 nnUNet_compile=False
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$CODE"
sha256sum -c source.sha256
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=10 150 "$PYTHON" -u judge_repeatability.py \
 --case "$BASE/case_100430_00001" \
 --producer "$BASE/results_3296011_100430_00001" \
 --models /u/asanjeev/compression_probe_smalloutputs/judge_stage_fold4_v1/models \
 --output "$CODE/results_${SLURM_JOB_ID}"
test -f "$CODE/results_${SLURM_JOB_ID}/completion.json"
