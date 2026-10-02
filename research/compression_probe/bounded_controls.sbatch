#!/bin/bash
# Submit separately with --export=ALL,CONTROL_MODE=insertion|split4|posterior.
# Three jobs at10min each x interactive2x <=1 charged GPU-hour TOTAL.
# Hard approved round cap2 charged hours; no retry/requeue or expansion here.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --job-name=vae_control
#SBATCH --output=/projects/bdyo/asanjeev/compression_probe_20261001/control_%j.log
set -euo pipefail
ROOT=/projects/bdyo/asanjeev/compression_probe_20261001
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
case "${CONTROL_MODE:-}" in insertion|split4|posterior) ;; *) exit 2;; esac
export PYTHONPATH=$ROOT/wheels/monai-1.5.1-py3-none-any.whl:$ROOT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
sha256sum -c controls_source.sha256
timeout 15 nvidia-smi -L
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=10 540 "$PYTHON" -u "$ROOT/bounded_controls.py" \
 --root "$ROOT" --mode "$CONTROL_MODE" --output "$ROOT/control_${CONTROL_MODE}_${SLURM_JOB_ID}"
