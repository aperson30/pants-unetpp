#!/bin/bash
# One bounded full-context raw-head diagnostic, not clinical detection.
# At billing2000: max0.333334chargedGPUh, including startup/failures.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --job-name=difftumor_raw_head
#SBATCH --output=/u/asanjeev/compression_probe_smalloutputs/difftumor_probe_v1/raw_head_%j.log
set -euo pipefail
umask 077
CODE=/u/asanjeev/compression_probe_smalloutputs/difftumor_probe_v1
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export PYTHONPATH=/projects/bdyo/asanjeev/compression_probe_20261001/wheels/monai-1.5.1-py3-none-any.whl:$CODE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$CODE"
sha256sum -c source.sha256
test "$(df --output=avail -B1 "$CODE" | tail -1)" -ge 2000000000
MANIFEST_SHA=$(sha256sum inputs.json | cut -d' ' -f1)
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
 timeout --signal=TERM --kill-after=10 560 "$PYTHON" -u difftumor_probe.py \
 --mode run --manifest inputs.json --manifest-sha "$MANIFEST_SHA" \
 --output "$CODE/results_${SLURM_JOB_ID}"
test -f "$CODE/results_${SLURM_JOB_ID}/completion.json"
