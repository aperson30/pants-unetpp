#!/bin/bash
# Encode once, mean plus three fixed-seed samples; report all, no retry.
# Maximum0.2charge-equivalentGPUh at billing2000.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=00:06:00
#SBATCH --no-requeue
#SBATCH --job-name=posterior_control
#SBATCH --output=/u/asanjeev/compression_probe_smalloutputs/posterior_control_v1/posterior_%j.log
set -euo pipefail
CODE=/u/asanjeev/compression_probe_smalloutputs/posterior_control_v1
ROOT=/projects/bdyo/asanjeev/compression_probe_20261001
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export PYTHONPATH=/u/asanjeev/compression_probe_smalloutputs/paired_judge_replication_small_v2/deps:$ROOT/wheels/monai-1.5.1-py3-none-any.whl:$CODE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 nnUNet_compile=False
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$CODE"
sha256sum -c source.sha256
test "$(df --output=avail -B1 "$CODE" | tail -1)" -ge 4000000000
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=10 330 "$PYTHON" -u posterior_control.py \
 --case /u/asanjeev/compression_probe_smalloutputs/paired_judge_pilot_v1/case \
 --producer /u/asanjeev/compression_probe_smalloutputs/paired_judge_pilot_v2/results_3295395 \
 --models /u/asanjeev/compression_probe_smalloutputs/judge_stage_fold4_v1/models \
 --vae "$ROOT/inputs/autoencoder_v1.pt" \
 --vae-sha 1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3 \
 --output "$CODE/results_${SLURM_JOB_ID}"
test -f "$CODE/results_${SLURM_JOB_ID}/completion.json"
