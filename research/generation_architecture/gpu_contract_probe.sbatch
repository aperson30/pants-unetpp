#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --job-name=nested_denoiser_contract
#SBATCH --output=/u/asanjeev/generation_architecture_20261008_v1/probe_%j.log
set -euo pipefail
cd /u/asanjeev/generation_architecture_20261008_v1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=""
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
  timeout --signal=TERM --kill-after=5 105 \
  /projects/bdyo/asanjeev/pants_venv/bin/python -u gpu_contract_probe.py
