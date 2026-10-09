#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --job-name=architecture_learning_pilot
#SBATCH --output=/u/asanjeev/generation_architecture_20261008_v1/learning_pilot_%j.log
set -euo pipefail
cd /u/asanjeev/generation_architecture_20261008_v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Explicit approved total cap 0.5h; this job's 10min factor-two maximum is 1/3h,
# before provider rounding. No automatic retry or queue racing.
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
  timeout --signal=TERM --kill-after=5 550 \
  ./baseline_venv/bin/python -u architecture_learning_pilot.py \
  --data-dir /u/asanjeev/generation_architecture_20261008_v1/architecture_pilot_inputs_v2 \
  --out-dir "/u/asanjeev/generation_architecture_20261008_v1/learning_pilot_${SLURM_JOB_ID}" \
  --charged-cap 0.5
