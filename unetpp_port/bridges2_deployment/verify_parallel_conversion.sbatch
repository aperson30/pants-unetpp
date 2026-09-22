#!/bin/bash
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=00:30:00
#SBATCH --job-name=pants_convert_parity
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/logs/convert_parity_%j.log

set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
REPO=$ROOT/repo
VENV=$ROOT/venv
SOURCE=$ROOT/calibration/source
SERIAL=$ROOT/calibration/nnUNet_raw/Dataset001_PanTSCalib
PARALLEL=$ROOT/calibration/parallel_parity/Dataset001_PanTSCalib
ANSWER=$ROOT/calibration/parallel_parity/empty_test_answer_key

test -f "$SERIAL/dataset.json"
rm -rf "$ROOT/calibration/parallel_parity"
mkdir -p "$ANSWER"
"$VENV/bin/python" "$REPO/data_conversion/convert_pants_to_nnunet.py" \
  --pants-root "$SOURCE" --nnunet-dataset-dir "$PARALLEL" \
  --test-answer-key-dir "$ANSWER" --workers 6
"$VENV/bin/python" "$REPO/unetpp_port/bridges2_deployment/verify_parallel_conversion.py" \
  --serial "$SERIAL" --parallel "$PARALLEL"
rm -rf "$ROOT/calibration/parallel_parity"
echo BRIDGES2_PARALLEL_CONVERSION_VERIFIED
