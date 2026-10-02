#!/bin/bash
# One GPU, seven minutes max, interactive 2x: <=0.23334 charge-hours.
# Separate additional 0.25 charge-hour cap; NO retries/requeue.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:07:00
#SBATCH --no-requeue
#SBATCH --job-name=msd_small_vae
#SBATCH --output=/projects/bdyo/asanjeev/compression_probe_20261001/small_probe_%j.log
set -euo pipefail
ROOT=/projects/bdyo/asanjeev/compression_probe_20261001
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export PYTHONPATH=$ROOT/wheels/monai-1.5.1-py3-none-any.whl:$ROOT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=$ROOT/mplconfig
cd "$ROOT"
sha256sum -c source.sha256
sha256sum -c small_source.sha256
nvidia-smi -L
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=10 380 "$PYTHON" -u "$ROOT/run_maisi.py" \
 --manifest "$ROOT/small_followup_120/inputs_manifest.json" \
 --output "$ROOT/small_results_${SLURM_JOB_ID}" --max-cases 1
