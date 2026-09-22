#!/bin/bash
# Hard budget cap: one H100 for six minutes = 0.1 H100 GPU-hour.
#SBATCH --account=cis260296p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=00:06:00
#SBATCH --job-name=pants_h100_gate
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/logs/h100_gate_%j.log

set -euo pipefail
PROJECT_ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
REPO=$PROJECT_ROOT/repo
VENV=$PROJECT_ROOT/venv
CALIB=$PROJECT_ROOT/calibration
RESULT=$CALIB/h100_calibration_${SLURM_JOB_ID}.json

mkdir -p "$PROJECT_ROOT/logs"
echo "=== Bridges-2 bounded H100 gate ==="
hostname
date -u
git -C "$REPO" rev-parse HEAD
nvidia-smi -L

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export nnUNet_raw=$CALIB/nnUNet_raw
export nnUNet_preprocessed=$CALIB/nnUNet_preprocessed
export nnUNet_results=$CALIB/nnUNet_results
export nnUNet_n_proc_DA=8
export nnUNet_compile=true
export TORCHINDUCTOR_CACHE_DIR="${LOCAL:-/tmp}/pants_h100_gate_${SLURM_JOB_ID}"
mkdir -p "$nnUNet_results" "$TORCHINDUCTOR_CACHE_DIR"

TRAINER_ARGS=()
if [ -n "${CALIBRATION_TRAINERS:-}" ]; then
    read -r -a REQUESTED_TRAINERS <<< "$CALIBRATION_TRAINERS"
    TRAINER_ARGS=(--trainers "${REQUESTED_TRAINERS[@]}")
fi

"$VENV/bin/python" -c "import torch; assert torch.__version__.startswith('2.10.0+cu126'), torch.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"
test -f "$nnUNet_preprocessed/Dataset001_PanTSCalib/nnUNetPlansBS4.json"

# Leave 30 seconds for Python shutdown/log flush before Slurm's absolute six-minute limit.
timeout --signal=TERM --kill-after=10s 330s \
  "$VENV/bin/python" "$REPO/unetpp_port/bridges2_deployment/h100_calibration.py" \
    --plans "$nnUNet_preprocessed/Dataset001_PanTSCalib/nnUNetPlansBS4.json" \
    --dataset-json "$nnUNet_raw/Dataset001_PanTSCalib/dataset.json" \
    --output "$RESULT" --warmup 1 --steps 3 "${TRAINER_ARGS[@]}"

test -s "$RESULT"
"$VENV/bin/python" -c "import json, os; p=json.load(open('$RESULT')); expected=len(os.environ.get('CALIBRATION_TRAINERS', '').split()) or 4; assert p['ok'] is True; assert len(p['results']) == expected"
echo BRIDGES2_H100_GATE_DONE
