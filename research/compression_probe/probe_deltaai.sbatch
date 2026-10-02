#!/bin/bash
# One GPU, <=5 min; 2x charge => <=0.167 hours, plus attempt1 0.0573.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:05:00
#SBATCH --job-name=msd_vae_probe
#SBATCH --output=/projects/bdyo/asanjeev/compression_probe_20261001/probe_%j.log
set -euo pipefail
ROOT=/projects/bdyo/asanjeev/compression_probe_20261001
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export PYTHONPATH=$ROOT/wheels/monai-1.5.1-py3-none-any.whl:$ROOT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=$ROOT/mplconfig
test -f "$ROOT/CPU_PREFLIGHT_OK"
cd "$ROOT"
sha256sum -c source.sha256
nvidia-smi -L
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=10 260 "$PYTHON" -u "$ROOT/run_maisi.py" \
 --manifest "$ROOT/inputs_manifest.json" --output "$ROOT/results_${SLURM_JOB_ID}" --max-cases 1
