#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:03:00
#SBATCH --no-requeue
#SBATCH --job-name=smile_component_smoke
#SBATCH --output=/u/asanjeev/generation_architecture_20261008_v1/baseline_smoke_%j.log
set -euo pipefail
cd /u/asanjeev/generation_architecture_20261008_v1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
  timeout --signal=TERM --kill-after=5 160 \
  ./baseline_venv/bin/python -u baseline_gpu_smoke.py
