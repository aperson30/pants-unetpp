#!/bin/bash
# Prepared, not automatically submitted. Regular billing1000 must be verified
# on the held job:18min x1GPU x1 charge <=0.30 charged-hours. No retries.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:18:00
#SBATCH --no-requeue
#SBATCH --job-name=vae_contrast
#SBATCH --output=/projects/bdyo/asanjeev/compression_probe_20261001/contrast_%j.log
set -euo pipefail
ROOT=/projects/bdyo/asanjeev/compression_probe_20261001
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export PYTHONPATH=$ROOT/wheels/monai-1.5.1-py3-none-any.whl:$ROOT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
sha256sum -c contrast_source.sha256
timeout 15 nvidia-smi -L
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=10 1020 "$PYTHON" -u "$ROOT/contrast_probe.py" \
 --root "$ROOT" --output "$ROOT/contrast_results_${SLURM_JOB_ID}"
