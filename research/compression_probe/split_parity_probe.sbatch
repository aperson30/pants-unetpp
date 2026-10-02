#!/bin/bash
# One already-tested public case, runtime-only split parity. No automatic retry.
# Max0.166667chargedGPUh, included in existing2h campaign.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --job-name=maisi_split_parity
#SBATCH --output=/u/asanjeev/compression_probe_smalloutputs/split_parity_v1/split_%j.log
set -euo pipefail
CODE=/u/asanjeev/compression_probe_smalloutputs/split_parity_v1
ROOT=/projects/bdyo/asanjeev/compression_probe_20261001
export PYTHONPATH=/u/asanjeev/compression_probe_smalloutputs/paired_judge_pilot_v1/deps:$ROOT/wheels/monai-1.5.1-py3-none-any.whl:$CODE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"
sha256sum -c source.sha256
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=10 270 /projects/bdyo/asanjeev/pants_venv/bin/python -u split_parity_probe.py
