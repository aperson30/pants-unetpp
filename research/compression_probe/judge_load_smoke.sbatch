#!/bin/bash
# Single model compatibility gate, NOT clinical inference. No automatic retry.
# Submit HELD first; verify billing/account/resource settings before release.
# At billing1000, five-minute maximum costs <=0.083334 charged GPU-hour.
# If moved while held to ghx4-interactive, reserve conservative2x <=0.166667.
# Job3294898 was moved explicitly; account/resource/partition recorded in status.
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --job-name=judge_load_gate
#SBATCH --output=/u/asanjeev/compression_probe_smalloutputs/judge_load_gate_v1/load_%j.log
set -euo pipefail
CODE=/u/asanjeev/compression_probe_smalloutputs/judge_load_gate_v1
MODELS=/u/asanjeev/compression_probe_smalloutputs/judge_stage_fold4_v1/models
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"
sha256sum -c source.sha256
test -f "$MODELS/completion.json"
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
  timeout --signal=TERM --kill-after=10 240 "$PYTHON" -u "$CODE/judge_load_smoke.py" \
  --models "$MODELS" --output "$CODE/result_${SLURM_JOB_ID}.json"
