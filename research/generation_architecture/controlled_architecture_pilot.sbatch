#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:06:00
#SBATCH --no-requeue
#SBATCH --job-name=controlled_architecture_pilot
#SBATCH --output=/u/asanjeev/generation_architecture_20261008_v1/controlled_pilot_%j.log
set -euo pipefail
cd /u/asanjeev/generation_architecture_20261008_v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Maximum additional0.2 charged h at factor2, cumulative<=~0.23278 before rounding.
# Internal300s deadline; outer330s; allocation360s. No retries.
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=5 330 \
 ./baseline_venv/bin/python -u controlled_architecture_pilot.py \
 --root /u/asanjeev/generation_architecture_20261008_v1 \
 --out-dir "controlled_pilot_${SLURM_JOB_ID}"
